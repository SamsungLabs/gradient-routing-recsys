import os
from logging import getLogger

import hydra
import torch
from clearml import Task, TaskTypes
from dotenv import load_dotenv
from omegaconf import DictConfig
from ray import tune
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import HyperTuning, Trainer
from recbole.utils import FeatureSource, FeatureType, init_logger, init_seed

from gradient_routing_recsys import get_project_root, import_class
from gradient_routing_recsys.datasets import SubModuleHierarchicalRecommenderDataset

load_dotenv()


def objective_function(config_dict=None, cfg_hydra: DictConfig = None, saved=True):
    keys_to_remove = []
    for key in config_dict.keys():
        if ":" in key and key.split(":")[0] in ("user_pref_net", "recommender_net"):
            key1, key2 = key.split(":")
            cfg_hydra[key1]["config"][key2] = config_dict[key]
            keys_to_remove.append(key)

    for key in keys_to_remove:
        config_dict.pop(key)

    config = Config(
        model=import_class(cfg_hydra["model"]["_target_"]),
        dataset=cfg_hydra["dataset"]["name"],
        config_dict=config_dict,
        config_file_list=[
            os.path.join(
                get_project_root(), cfg_hydra["dataset"]["recbole_config_file"]
            )
        ],
    )
    config.final_config_dict["data_path"] = os.path.join(
        get_project_root(), config.final_config_dict["data_path"]
    )

    # Set device to the specified GPU directly
    gpu_id = cfg_hydra.get("gpu_id", 0)
    config["device"] = torch.device(f"cuda:{gpu_id}")

    # Set worker configuration
    worker = cfg_hydra.get("worker", 0)
    config["worker"] = worker

    init_seed(config["seed"], config["reproducibility"])
    logger = getLogger()
    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)
    init_logger(config)

    # Log GPU and worker information
    logger.info("=" * 50)
    logger.info(f"GPU Configuration")
    logger.info(f"  GPU ID: {gpu_id}")
    logger.info(f"  Device: {config['device']}")
    logger.info(f"Worker Configuration")
    logger.info(f"  Worker: {worker}")
    logger.info("=" * 50)

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    additional_args = dict()
    if cfg_hydra["model"]["_target_"].split(".")[-1] == "HierarchicalRecommender":
        # Update device in subnet configs to match the selected GPU
        device_str = f"cuda:{gpu_id}"
        if cfg_hydra["user_pref_net"].get("config"):
            cfg_hydra["user_pref_net"]["config"]["device"] = device_str
        if cfg_hydra["recommender_net"].get("config"):
            cfg_hydra["recommender_net"]["config"]["device"] = device_str

        user_pref_net_dataset = SubModuleHierarchicalRecommenderDataset.cast(
            dataset,
            cat_feat_field=cfg_hydra["dataset"]["cat_feat_field"],
            cols_to_remove=cfg_hydra["dataset"]["user_pref_net"]["cols_to_remove"],
            cols_to_add=[("cat_id", FeatureType.TOKEN, FeatureSource.ITEM_ID)],
        )
        recommender_net_dataset = SubModuleHierarchicalRecommenderDataset.cast(
            dataset,
            cat_feat_field=cfg_hydra["dataset"]["cat_feat_field"],
            cols_to_remove=(
                cfg_hydra["dataset"]["recommender_net"]["cols_to_remove"]
                if cfg_hydra["model"]["input_user_fields_rec_net_flag"]
                else cfg_hydra["dataset"]["recommender_net"]["cols_to_remove"]
                + cfg_hydra["dataset"]["user_feat_cols"]
            ),
            cols_to_add=[("user_prefs", FeatureType.FLOAT_SEQ, FeatureSource.USER)],
        )
        user_pref_net = hydra.utils.instantiate(
            cfg_hydra["user_pref_net"], dataset=user_pref_net_dataset
        )
        recommender_net = hydra.utils.instantiate(
            cfg_hydra["recommender_net"],
            dataset=recommender_net_dataset,
        )
        additional_args["user_pref_net"] = user_pref_net
        additional_args["recommender_net"] = recommender_net
        additional_args["cat_feat_field"] = cfg_hydra["dataset"]["cat_feat_field"]

        # model loading and initialization
        model = hydra.utils.instantiate(
            cfg_hydra["model"],
            recbole_config=config,
            dataset=train_data.dataset,
            **additional_args,
        ).to(config["device"])
    else:
        model = hydra.utils.instantiate(
            cfg_hydra["model"],
            config,
            train_data.dataset,
        ).to(config["device"])

    model_name = config["model"]
    logger.info(f"Model name: {model_name}")
    logger.info(f"Using GPU: {config['device']}")
    trainer = Trainer(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved
    )
    test_result = trainer.evaluate(test_data, load_best_model=saved)
    tune.report(**test_result)

    return {
        "model": model_name,
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }


@hydra.main(
    config_path=os.path.join(get_project_root(), "configs", "hydra"),
    config_name="config.yaml",
    version_base=None,
)
def main(cfg_hydra: DictConfig):
    # Set GPU from config
    gpu_id = cfg_hydra.get("gpu_id", 0)

    # Validate GPU exists
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA is not available on this system. Cannot use GPU {gpu_id}. "
            f"Please check your CUDA installation or set gpu_id to use CPU (not recommended for training)."
        )

    num_gpus = torch.cuda.device_count()
    if gpu_id >= num_gpus:
        raise RuntimeError(
            f"GPU ID {gpu_id} does not exist on this system. "
            f"Available GPUs: {num_gpus} (IDs: 0-{num_gpus-1}). "
            f"Please set gpu_id to a valid GPU ID in the range [0, {num_gpus-1}]."
        )

    logger = getLogger()
    logger.info("=" * 50)
    logger.info("GPU Validation Successful")
    logger.info(f"  Requested GPU ID: {gpu_id}")
    logger.info(f"  Total GPUs available: {num_gpus}")
    logger.info(f"  GPU Name: {torch.cuda.get_device_name(gpu_id)}")
    logger.info("=" * 50)

    # Determine parameter file and tags based on model configuration
    if cfg_hydra["model"]["_target_"].split(".")[-1] == "HierarchicalRecommender":
        params_file = "hierarchical_recommender/"

        if cfg_hydra["user_pref_net"]["_target_"].split(".")[-1] == "WideDeep":
            params_file += "wide_deep-"
            tag = "HierarchicalRecommender.u=WideDeep"
        elif cfg_hydra["user_pref_net"]["_target_"].split(".")[-1] == "LR":
            params_file += "lr-"
            tag = "HierarchicalRecommender.u=LR"
        elif cfg_hydra["user_pref_net"]["_target_"].split(".")[-1] == "DeepFM":
            params_file += "deepfm-"
            tag = "HierarchicalRecommender.u=DeepFM"
        else:
            raise ValueError("Hyperparameter tuning not supported for this submodel")

        if cfg_hydra["recommender_net"]["_target_"].split(".")[-1] == "WideDeep":
            params_file += "wide_deep.params"
            tag += ".WideDeep"
        elif cfg_hydra["recommender_net"]["_target_"].split(".")[-1] == "LR":
            params_file += "lr.params"
            tag += ".LR"
        elif cfg_hydra["recommender_net"]["_target_"].split(".")[-1] == "DeepFM":
            params_file += "deepfm.params"
            tag += ".DeepFM"
        elif cfg_hydra["recommender_net"]["_target_"].split(".")[-1] == "FiGNN":
            params_file += "fignn.params"
            tag += ".FiGNN"
        else:
            raise ValueError("Hyperparameter tuning not supported for this submodel")

        if cfg_hydra.model.input_user_fields_rec_net_flag:
            tag += ".user_flag"
    elif cfg_hydra["model"]["_target_"].split(".")[-1] == "WideDeep":
        params_file = "wide_deep.params"
        tag = "WideDeep"
    elif cfg_hydra["model"]["_target_"].split(".")[-1] == "LR":
        params_file = "lr.params"
        tag = "LR"
    elif cfg_hydra["model"]["_target_"].split(".")[-1] == "DeepFM":
        params_file = "deepfm.params"
        tag = "DeepFM"
    elif cfg_hydra["model"]["_target_"].split(".")[-1] == "FiGNN":
        params_file = "fignn.params"
        tag = "FiGNN"
    else:
        raise ValueError("Hyperparameter tuning not supported for this model")

    # Initialize ClearML task
    task = Task.init(
        project_name=f"{cfg_hydra.clearml.project_name}/{cfg_hydra['model']['_target_'].split('.')[-1]}",
        task_name="Hparams_tuning",
        task_type=TaskTypes.optimizer,
        tags=[str(i) for i in tag.split(".")]
        + [
            f"{cfg_hydra['dataset']['recbole_config_file'].split('/')[-1].split('.')[0]}"
        ],
        reuse_last_task_id=False,
    )

    hp = HyperTuning(
        objective_function,
        algo=cfg_hydra.hparams_tuning.algo,
        early_stop=cfg_hydra.hparams_tuning.early_stop,
        max_evals=cfg_hydra.hparams_tuning.max_evals,
        params_file=os.path.join(get_project_root(), "hparams", params_file),
        fixed_config_file_list=cfg_hydra,
        display_file=cfg_hydra.hparams_tuning.display_file,
    )
    hp.run()
    hp.export_result(
        output_file=os.path.join(
            get_project_root(), cfg_hydra.hparams_tuning.output_file
        )
    )
    print("best params: ", hp.best_params)
    print("best result: ")
    print(hp.params2result[hp.params2str(hp.best_params)])


if __name__ == "__main__":
    main()
