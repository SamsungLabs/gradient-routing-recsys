"""
Script to prepare ML-32M dataset with user preference vectors.
This script adds user_pref_vector_implicit and user_pref_vector_explicit columns.
"""

from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm


def prepare_ml32m_dataset(inter_path: Path, item_path: Path, output_path: Path):
    """
    Process ML-32M interaction data and add user preference vectors.

    Args:
        inter_path: Path to interaction file (ml-32m.inter)
        item_path: Path to item file (ml-32m.item)
        output_path: Path to save processed interactions
    """
    print(f"Loading interaction data from {inter_path}")
    inter_df = pl.read_csv(
        inter_path,
        separator="\t",
        has_header=True,
        new_columns=["user_id", "item_id", "rating", "timestamp"],
        schema_overrides={
            "user_id": pl.Int64,
            "item_id": pl.Int64,
            "rating": pl.Float32,
            "timestamp": pl.Int64,
        },
    )

    print(f"Loading item data from {item_path}")
    item_df = pl.read_csv(
        item_path,
        separator="\t",
        has_header=True,
        new_columns=["item_id", "movie_name", "release_year", "type"],
    )

    print("Merging interaction and item data...")
    inter_df_2 = inter_df.join(item_df, on="item_id", how="left")

    print("Sorting by user_id and timestamp...")
    sorted_df = inter_df_2.sort(["user_id", "timestamp"])

    print("Extracting categories and creating mapping...")
    all_categories = set()
    for type_str in item_df["type"].to_list():
        categories = str(type_str).split()
        all_categories.update(categories)

    category_to_idx = {t: i for i, t in enumerate(all_categories)}
    num_categories = len(all_categories)
    print(f"Found {num_categories} unique categories")

    print("Splitting type column into categories...")
    sorted_df = sorted_df.with_columns(
        pl.col("type").str.split(" ").alias("categories")
    )

    # Add preference vectors for implicit mode
    print("Calculating implicit user preference vectors...")
    df = prepare_user_preference(
        sorted_df, num_categories, category_to_idx, mode="implicit"
    )

    # Add preference vectors for explicit mode
    print("Calculating explicit user preference vectors...")
    df = prepare_user_preference(
        df, num_categories, category_to_idx, mode="explicit", thr_explicit=4.0
    )

    print("Selecting and renaming columns...")
    columns = list(inter_df.columns) + [
        "user_pref_vector_implicit",
        "user_pref_vector_explicit",
    ]
    df = df[columns]

    df = df.rename(
        {
            "user_id": "user_id:token",
            "item_id": "item_id:token",
            "rating": "rating:float",
            "timestamp": "timestamp:float",
            "user_pref_vector_implicit": "user_pref_vector_implicit:float_seq",
            "user_pref_vector_explicit": "user_pref_vector_explicit:float_seq",
        }
    )

    print(f"Saving processed data to {output_path}")
    df.write_csv(output_path, separator="\t", include_header=True, quote_style="never")

    print(f"Dataset preparation complete! Processed {len(df)} interactions.")


def prepare_user_preference(
    df, num_categories, category_to_idx, mode="implicit", thr_explicit=4.0
):
    """
    Optimized Polars version that balances vectorization with necessary iteration.

    Args:
        df: Polars DataFrame with columns [user_id, categories, rating]
        num_categories: Number of category types
        category_to_idx: Dictionary mapping category names to indices
        mode: 'implicit' or 'explicit' mode
        thr_explicit: Rating threshold for explicit mode (default: 4.0)

    Returns:
        Polars DataFrame with added user_pref_vector_{mode} column
    """
    # Ensure we have a Polars DataFrame
    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df)

    print(f"Calculating user preference vectors (mode={mode})...")

    # Create uniform distribution
    uniform_pref = np.ones(num_categories) / num_categories

    # For explicit mode, filter rows that won't contribute to counts
    if mode == "explicit":
        df_with_contribution = df.with_columns(
            [(pl.col("rating") >= thr_explicit).alias("contributes")]
        )
    else:
        df_with_contribution = df.with_columns([pl.lit(True).alias("contributes")])

    # Mark first interaction per user
    df_with_contribution = df_with_contribution.with_columns(
        [pl.col("user_id").cum_count().over("user_id").alias("interaction_rank")]
    )

    # Convert to list of dictionaries for efficient iteration
    rows = df_with_contribution.to_dicts()

    # Track user preference counts and seen users
    user_pref_counts = {}
    seen_users = set()
    preference_vectors = []

    # Iterate through rows
    for row in tqdm(rows, total=len(rows)):
        user_id = row["user_id"]
        categories = row["categories"]
        contributes = row["contributes"]
        is_first = row["interaction_rank"] == 1

        if is_first:
            # First interaction: uniform distribution
            pref = uniform_pref.tolist()
            seen_users.add(user_id)
            user_pref_counts[user_id] = np.zeros(num_categories, dtype=np.int32)
        else:
            # Subsequent interactions: based on cumulative history
            counts = user_pref_counts[user_id].copy()
            total = counts.sum()

            if total > 0:
                # Normalize to sum to 1
                pref = (counts.astype(np.float32) / total).tolist()
            else:
                pref = uniform_pref.tolist()

        # Convert to space-separated string with exactly 4 decimal places
        preference_vectors.append(" ".join(f"{v:.4f}" for v in pref))

        # Update counts with current interaction's categories
        if contributes:
            for cat in categories:
                if cat in category_to_idx:
                    cat_idx = category_to_idx[cat]
                    user_pref_counts[user_id][cat_idx] += 1

    # Add preference vectors to DataFrame
    df = df.with_columns([pl.Series(f"user_pref_vector_{mode}", preference_vectors)])

    return df


if __name__ == "__main__":
    # Get dataset paths
    dataset_dir = Path("dataset/ml-32m")

    inter_path = dataset_dir / "ml-32m.inter"
    item_path = dataset_dir / "ml-32m.item"
    output_path = dataset_dir / "ml-32m.inter"  # Overwrite original

    prepare_ml32m_dataset(inter_path, item_path, output_path)
