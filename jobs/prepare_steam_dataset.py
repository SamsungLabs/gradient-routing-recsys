"""
Script to prepare Steam dataset with user preference vectors.
This script adds user_pref_vector_implicit column for implicit feedback.
"""

from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm


def prepare_steam_dataset(inter_path: Path, item_path: Path, output_path: Path):
    """
    Process Steam interaction data and add user preference vectors.

    Args:
        inter_path: Path to interaction file (steam.inter)
        item_path: Path to item file (steam.item)
        output_path: Path to save processed interactions
    """
    print(f"Loading interaction data from {inter_path}")
    inter_df = pl.read_csv(
        inter_path,
        separator="\t",
        has_header=True,
    )
    # Store original column names for later restoration
    original_inter_columns = list(inter_df.columns)
    # Rename columns to remove type suffixes
    rename_map = {}
    for col in inter_df.columns:
        if ":" in col:
            rename_map[col] = col.split(":")[0]
    inter_df = inter_df.rename(rename_map)
    inter_df = inter_df.with_columns(pl.col("product_id").cast(pl.String))

    print(f"Loading item data from {item_path}")
    item_df = pl.read_csv(
        item_path,
        separator="\t",
        has_header=True,
    )
    # Build rename mapping from actual column names
    rename_map = {}
    for col in item_df.columns:
        if ":" in col:
            rename_map[col] = col.split(":")[0]
    item_df = item_df.rename(rename_map)
    item_df = item_df.with_columns(pl.col("product_id").cast(pl.String))

    print("Merging interaction and item data...")
    inter_df_2 = inter_df.join(item_df, on="product_id", how="left")

    print("Sorting by user_id and timestamp...")
    sorted_df = inter_df_2.sort(["user_id", "timestamp"])

    print("Extracting genres and counting frequency...")
    from collections import Counter

    category_counts = Counter()
    for cat_str in item_df["genres"].to_list():
        # Genres are comma-separated in Steam dataset
        categories = str(cat_str).split(", ")
        for cat in categories:
            # Clean up the category string
            cat_clean = cat.strip().strip("'").strip('"').strip()
            if cat_clean:
                category_counts[cat_clean] += 1

    # Use all unique genres
    all_categories = [cat for cat, _ in category_counts.most_common()]
    print(f"Total unique genres: {len(category_counts)}")
    print(f"Using all genres: {all_categories}")

    # Create mapping for all genres
    category_to_idx = {cat: i for i, cat in enumerate(all_categories)}
    num_categories = len(all_categories)
    print(f"Total categories: {num_categories}")

    print("Splitting genres column into list...")
    sorted_df = sorted_df.with_columns(
        pl.col("genres")
        .str.split(", ")
        .map_elements(
            lambda lst: [
                c.strip().strip("'").strip('"').strip() for c in lst if c.strip()
            ],
            return_dtype=pl.List(pl.String),
        )
        .alias("genres_list")
    )

    # Add preference vectors for implicit mode (Steam is implicit feedback)
    print("Calculating implicit user preference vectors...")
    df = prepare_user_preference(
        sorted_df, num_categories, category_to_idx, mode="implicit"
    )

    print("Selecting and renaming columns...")
    columns = list(inter_df.columns) + [
        "user_pref_vector_implicit",
    ]
    df = df[columns]

    # Restore original column names with type suffixes
    rename_back = {}
    for original_col in original_inter_columns:
        if ":" in original_col:
            base_name = original_col.split(":")[0]
            rename_back[base_name] = original_col
    rename_back["user_pref_vector_implicit"] = "user_pref_vector_implicit:float_seq"
    df = df.rename(rename_back)

    print(f"Saving processed data to {output_path}")
    df.write_csv(output_path, separator="\t", include_header=True, quote_style="never")

    print(f"Dataset preparation complete! Processed {len(df)} interactions.")


def prepare_user_preference(df, num_categories, category_to_idx, mode="implicit"):
    """
    Optimized Polars version that balances vectorization with necessary iteration.

    Args:
        df: Polars DataFrame with columns [user_id, genres_list]
        num_categories: Number of category types
        category_to_idx: Dictionary mapping category names to indices
        mode: 'implicit' mode only for Steam dataset

    Returns:
        Polars DataFrame with added user_pref_vector_{mode} column
    """
    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df)

    print(f"Calculating user preference vectors (mode={mode})...")

    # Create uniform distribution
    uniform_pref = np.ones(num_categories) / num_categories

    df_with_contribution = df.with_columns([pl.lit(True).alias("contributes")])
    df_with_contribution = df_with_contribution.with_columns(
        [pl.col("user_id").cum_count().over("user_id").alias("interaction_rank")]
    )
    rows = df_with_contribution.to_dicts()

    user_pref_counts = {}
    seen_users = set()
    preference_vectors = []

    for row in tqdm(rows, total=len(rows)):
        user_id = row["user_id"]
        categories = row["genres_list"]
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

        preference_vectors.append(" ".join(f"{v:.4f}" for v in pref))

        if contributes and categories is not None:
            for cat in categories:
                if cat in category_to_idx:
                    cat_idx = category_to_idx[cat]
                    user_pref_counts[user_id][cat_idx] += 1

    df = df.with_columns([pl.Series(f"user_pref_vector_{mode}", preference_vectors)])

    return df


if __name__ == "__main__":
    dataset_dir = Path("dataset/steam")

    inter_path = dataset_dir / "steam.inter"
    item_path = dataset_dir / "steam.item"
    output_path = dataset_dir / "steam.inter"  # Overwrite original

    prepare_steam_dataset(inter_path, item_path, output_path)
