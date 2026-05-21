import os
from logging import getLogger
from pathlib import Path
from typing import Dict

import hydra
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import umap
from dotenv import load_dotenv
from omegaconf import DictConfig
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
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
    """Initialize HierarchicalRecommender and load weights from checkpoint."""

    # Initialize configuration
    config = Config(
        model=import_class(cfg_hydra["model"]["_target_"]),
        dataset=cfg_hydra["dataset"]["name"],
        config_file_list=[cfg_hydra["dataset"]["recbole_config_file"]],
    )
    init_seed(config["seed"], config["reproducibility"])

    # Logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # Dataset preparation
    dataset = create_dataset(config)
    logger.info(dataset)

    train_data, valid_data, test_data = data_preparation(config, dataset)

    # Initialize HierarchicalRecommender
    additional_args = dict()
    if cfg_hydra["model"]["_target_"].split(".")[-1] == "HierarchicalRecommender":
        # Prepare user preference network dataset
        user_pref_net_dataset = SubModuleHierarchicalRecommenderDataset.cast(
            dataset,
            cat_feat_field=cfg_hydra["dataset"]["cat_feat_field"],
            cols_to_remove=cfg_hydra["dataset"]["user_pref_net"]["cols_to_remove"],
            cols_to_add=[("cat_id", FeatureType.TOKEN, FeatureSource.ITEM_ID)],
        )

        # Prepare recommender network dataset
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

        # Instantiate subnetworks
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

        # Instantiate main model
        model = hydra.utils.instantiate(
            cfg_hydra["model"],
            recbole_config=config,
            dataset=train_data.dataset,
            **additional_args,
        ).to(config["device"])
    else:
        model = hydra.utils.instantiate(
            cfg_hydra["model"], config, train_data.dataset
        ).to(config["device"])

    logger.info(model)

    # Load saved weights from ClearML task "Experiment"
    from clearml import Task

    # Get the "Experiment" task from ClearML
    task = Task.get_task(
        project_name=f"{cfg_hydra.clearml.project_name}/{config.model}",
        task_name="Experiment",
    )

    if task is None:
        logger.error("Could not find 'Experiment' task in ClearML")
        raise FileNotFoundError("Could not find 'Experiment' task in ClearML")

    # Download the last artifact (checkpoint)
    logger.info("Downloading checkpoint from ClearML 'Experiment' task...")
    artifact = task.models["output"][-1]  # Get the last output model artifact

    # Download to local file
    checkpoint_local_path = artifact.get_local_copy()
    logger.info(f"Checkpoint downloaded to {checkpoint_local_path}")

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_local_path, map_location=config["device"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    logger.info("Model loaded successfully from ClearML")

    # Compute user_pref_net_softmax for each user
    logger.info("Computing user preference softmax values...")
    user_pref_softmax_dict = compute_user_pref_softmax(model, dataset, config)

    logger.info(f"Computed preferences for {len(user_pref_softmax_dict)} users")

    # Visualize user preference vectors using UMAP
    logger.info("Visualizing user preference vectors with UMAP...")
    visualize_with_umap(user_pref_softmax_dict, cfg_hydra["visualisation"])
    logger.info("Visualization completed and saved to ./visualizations")


def visualize_with_umap(
    user_pref_softmax_dict: Dict[int, torch.Tensor],
    cfg: DictConfig,
):
    """
    Visualize user preference vectors using UMAP dimensionality reduction.

    Parameters
    ----------
    user_pref_softmax_dict : Dict[int, torch.Tensor]
        Dictionary mapping user_id to user_pref_net_softmax tensor
    cfg : DictConfig
        Configuration dictionary containing visualization parameters
    """
    # Convert dictionary to numpy array
    user_ids = sorted(user_pref_softmax_dict.keys())
    vectors = np.stack(
        [user_pref_softmax_dict[user_id].numpy() for user_id in user_ids]
    )

    # logger.info(f"Input vectors shape: {vectors.shape}")

    # Apply UMAP for dimensionality reduction to 2D
    reducer = umap.UMAP(
        n_neighbors=cfg["umap"]["n_neighbors"],
        min_dist=cfg["umap"]["min_dist"],
        metric=cfg["umap"]["metric"],
        random_state=cfg["umap"]["random_state"],
    )
    embedding = reducer.fit_transform(vectors)

    # Get save directory from config
    save_dir = cfg["save_dir"]

    # Create save directory if it doesn't exist
    Path(save_dir).mkdir(exist_ok=True)

    # Create visualization without coloring
    plt.figure(figsize=(14, 10))

    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c="steelblue",
        alpha=0.6,
        s=20,
    )

    # plt.xlabel("UMAP Component 1", fontsize=12)
    # plt.ylabel("UMAP Component 2", fontsize=12)
    plt.title("User Preference Vectors", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = Path(save_dir) / "user_preferences_umap_uncolored.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Create fourth visualization colored by most-interacted type from dataset
    # logger.info("Calculating most-interacted types from dataset...")
    user_most_interacted_type = calculate_most_interacted_type(
        dataset_path="./dataset/ml-32m/",
        min_rating=cfg["min_rating"],
        fraction=cfg["fraction"],
        use_first=cfg["use_first"],
    )
    # logger.info(
    #    f"Calculated most-interacted types for {len(user_most_interacted_type)} users"
    # )

    # Map types to numeric values for coloring
    unique_types = sorted(set(user_most_interacted_type.values()))
    type_to_numeric = {type_val: idx for idx, type_val in enumerate(unique_types)}

    # Create array of type values for users in user_ids order
    user_types_numeric = []
    user_type_labels = []
    for user_id in user_ids:
        if user_id in user_most_interacted_type:
            user_types_numeric.append(
                type_to_numeric[user_most_interacted_type[user_id]]
            )
            user_type_labels.append(user_most_interacted_type[user_id])
        else:
            user_types_numeric.append(-1)
            user_type_labels.append("Unknown")

    user_types_numeric = np.array(user_types_numeric)

    # Filter out unknown users for better visualization
    valid_mask = user_types_numeric != -1

    plt.figure(figsize=(14, 10))

    plt.scatter(
        embedding[valid_mask, 0],
        embedding[valid_mask, 1],
        c=user_types_numeric[valid_mask],
        cmap="tab20",
        alpha=0.6,
        s=20,
    )

    # plt.colorbar(scatter4, label="Most-Interacted Type")
    # plt.xlabel("UMAP Component 1", fontsize=12)
    # plt.ylabel("UMAP Component 2", fontsize=12)
    plt.title(
        "User Preference Vectors (Colored by Most-Interacted Type)",
        fontsize=14,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path4 = Path(save_dir) / "user_preferences_umap_by_interacted_type.png"
    plt.savefig(save_path4, dpi=300, bbox_inches="tight")
    plt.close()

    # logger.info(f"Most-interacted type visualization saved to {save_path4}")


def calculate_most_interacted_type(
    dataset_path: str,
    min_rating: float = 4.0,
    fraction: float = 0.8,
    use_first: bool = True,
) -> Dict[int, str]:
    """
    Calculate the most-interacted type for each user based on interactions with rating >= min_rating.
    Only uses a fraction of each user's data (first or last).

    Parameters
    ----------
    dataset_path : str
        Path to the dataset directory
    min_rating : float, optional
        Minimum rating threshold (default: 4.0)
    fraction : float, optional
        Fraction of data to use (default: 0.8)
    use_first : bool, optional
        If True, use first fraction of data; if False, use last fraction (default: True)

    Returns
    -------
    Dict[int, str]
        Dictionary mapping user_id to most-interacted type
    """
    dataset_path = Path(dataset_path)

    # Load item data with polars
    item_df = pl.read_csv(dataset_path / "ml-32m.item", separator="\t")

    # Load interaction data with polars
    inter_df = pl.read_csv(dataset_path / "ml-32m.inter", separator="\t")

    # Filter by minimum rating
    high_rated_inter = inter_df.filter(pl.col("rating:float") >= min_rating)

    # Sort by timestamp (ascending order - oldest to newest)
    # if "timestamp:float" in high_rated_inter.columns
    high_rated_inter = high_rated_inter.sort("timestamp:float")
    # else:
    #     # Fallback: assume data is already in chronological order by row number
    #     high_rated_inter = high_rated_inter.with_row_index(name="row_num").sort(
    #         "row_num"
    #     )

    # Calculate row numbers per user to enable filtering
    high_rated_inter = high_rated_inter.with_columns(
        pl.col("user_id:token")
        .rank(method="ordinal")
        .over("user_id:token")
        .alias("user_row")
    )

    # Calculate total rows per user
    user_counts = high_rated_inter.group_by("user_id:token").agg(
        pl.len().alias("total_rows")
    )

    # Join the counts back
    high_rated_inter = high_rated_inter.join(
        user_counts, on="user_id:token", how="left"
    )

    # Calculate threshold for each user
    high_rated_inter = high_rated_inter.with_columns(
        (pl.col("total_rows") * fraction).alias("threshold")
    )

    # Filter to only use a fraction of each user's data (first or last by timestamp)
    # First = oldest interactions, Last = newest interactions
    if use_first:
        # Keep only rows where user_row <= threshold
        high_rated_inter = high_rated_inter.filter(
            pl.col("user_row") <= pl.col("threshold")
        )
    else:
        # Keep only rows where user_row > total_rows - threshold
        high_rated_inter = high_rated_inter.filter(
            pl.col("user_row") > (pl.col("total_rows") - pl.col("threshold"))
        )

    # Remove temporary columns
    high_rated_inter = high_rated_inter.drop(["user_row", "total_rows", "threshold"])
    if "row_num" in high_rated_inter.columns:
        high_rated_inter = high_rated_inter.drop("row_num")

    # Join with item data to get types
    high_rated_inter = high_rated_inter.join(
        item_df.select(["item_id:token", "type:token_seq"]),
        on="item_id:token",
        how="left",
    ).rename({"type:token_seq": "type"})

    # Remove rows where type couldn't be mapped
    high_rated_inter = high_rated_inter.filter(pl.col("type").is_not_null())

    # Split type string into individual genres and explode
    # "Comedy Drama Romance" -> ["Comedy", "Drama", "Romance"]
    high_rated_inter = high_rated_inter.with_columns(
        pl.col("type").str.split(" ").alias("genre")
    )

    # Explode to create separate rows for each genre
    high_rated_inter = high_rated_inter.explode("genre")

    # Remove empty genres
    high_rated_inter = high_rated_inter.filter(
        (pl.col("genre").is_not_null()) & (pl.col("genre") != "")
    )

    # Group by user and genre, count occurrences, and get most common genre
    user_genre_counts = (
        high_rated_inter.group_by(["user_id:token", "genre"])
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .group_by("user_id:token", maintain_order=True)
        .first()
        .to_dict(as_series=False)
    )

    # Convert to dictionary
    user_type_counts_dict = dict(
        zip(user_genre_counts["user_id:token"], user_genre_counts["genre"])
    )

    return user_type_counts_dict


def compute_user_pref_softmax(
    model: torch.nn.Module,
    dataset,
    config: Config,
) -> Dict[int, torch.Tensor]:
    """
    Compute user_pref_net_softmax for each user in the dataset.

    Parameters
    ----------
    model : torch.nn.Module
        The loaded HierarchicalRecommender model
    dataset
        The dataset object
    config : Config
        RecBole configuration

    Returns
    -------
    Dict[int, torch.Tensor]
        Dictionary mapping user_id to user_pref_net_softmax tensor
    """
    model.eval()
    device = config["device"]
    user_pref_softmax_dict = {}

    with torch.no_grad():
        # Get unique users from the dataset
        user_ids = dataset.user_num

        # Process users in batches
        batch_size = 1024

        for start_idx in range(1, user_ids + 1, batch_size):
            end_idx = min(start_idx + batch_size, user_ids + 1)
            current_batch_users = list(range(start_idx, end_idx))
            batch_size_current = len(current_batch_users)

            # Create interaction for the batch
            interaction_dict = {
                "user_id": torch.tensor(current_batch_users, device=device),
            }

            from recbole.data.interaction import Interaction

            interaction = Interaction(interaction_dict)

            # Prepare user preference network interaction
            user_pref_net_interaction = model._prepare_user_pref_net_interaction(
                interaction, batch_size_current
            )

            # Process through user preference network
            user_pref_net_output = model._process_user_pref_net(
                user_pref_net_interaction, batch_size_current
            )

            # Apply activation function
            user_pref_net_softmax = model._apply_activation_function(
                user_pref_net_output
            )

            # Store results for each user
            for i, user_id in enumerate(current_batch_users):
                user_pref_softmax_dict[user_id] = user_pref_net_softmax[i].cpu()

    return user_pref_softmax_dict


if __name__ == "__main__":
    main()
