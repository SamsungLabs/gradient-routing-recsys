import os
from logging import getLogger

import hydra
import torch
from clearml import Task, TaskTypes
from dotenv import load_dotenv
from omegaconf import DictConfig
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from recbole.utils import FeatureSource, FeatureType, init_logger, init_seed

from gradient_routing_recsys import get_project_root, import_class
from gradient_routing_recsys.datasets import SubModuleHierarchicalRecommenderDataset

load_dotenv()


@hydra.main(
    config_path=os.path.join(get_project_root(), "configs", "hydra"),
    config_name="config.yaml",
    version_base=None,
)
def main(cfg_hydra: DictConfig):
    config = Config(
        model=import_class(cfg_hydra["model"]["_target_"]),
        dataset=cfg_hydra["dataset"]["name"],
        config_file_list=[cfg_hydra["dataset"]["recbole_config_file"]],
    )

    # Override GPU and worker settings from Hydra config
    gpu_id = cfg_hydra.get("gpu_id", 0)
    config["device"] = torch.device(f"cuda:{gpu_id}")

    worker = cfg_hydra.get("worker", 0)
    config["worker"] = worker

    init_seed(config["seed"], config["reproducibility"])

    tags = [
        f"{config.model}",
        f"{cfg_hydra['dataset']['recbole_config_file'].split('/')[-1].split('.')[0]}",
    ]

    if cfg_hydra["model"]["_target_"].split(".")[-1] == "HierarchicalRecommender":
        tags += [
            f"{cfg_hydra.user_pref_net['_target_'].split('.')[-1]}-{cfg_hydra.recommender_net['_target_'].split('.')[-1]}",
        ]
        if cfg_hydra.model.input_user_fields_rec_net_flag:
            tags += ["user_flag"]

    task = Task.init(
        project_name=f"{cfg_hydra.clearml.project_name}/{config.model}",
        task_name="Experiment",
        task_type=TaskTypes.training,
        tags=tags,
        reuse_last_task_id=False,
    )

    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    additional_args = dict()
    if cfg_hydra["model"]["_target_"].split(".")[-1] == "HierarchicalRecommender":
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

    logger.info(model)

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    # model evaluation
    test_result = trainer.evaluate(test_data)

    logger.info("best valid result: {}".format(best_valid_result))
    logger.info("test result: {}".format(test_result))


if __name__ == "__main__":
    main()
