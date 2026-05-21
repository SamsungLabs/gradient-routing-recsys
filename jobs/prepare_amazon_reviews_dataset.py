"""
Script to prepare Amazon Reviews dataset with user preference vectors.
This script adds user_pref_vector_implicit and user_pref_vector_explicit columns.
"""

from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm


def prepare_amazon_dataset(inter_path: Path, item_path: Path, output_path: Path):
    """
    Process Amazon Reviews interaction data and add user preference vectors.

    Args:
        inter_path: Path to interaction file (Amazon_Video_Games.inter)
        item_path: Path to item file (Amazon_Video_Games.item)
        output_path: Path to save processed interactions
    """
    print(f"Loading interaction data from {inter_path}")
    inter_df = pl.read_csv(
        inter_path,
        separator="\t",
        has_header=True,
        new_columns=["user_id", "item_id", "rating", "timestamp"],
        schema_overrides={
            "user_id": pl.String,
            "item_id": pl.String,
            "rating": pl.Float32,
            "timestamp": pl.Int64,
        },
    )

    print(f"Loading item data from {item_path}")
    item_df = pl.read_csv(
        item_path,
        separator="\t",
        has_header=True,
        new_columns=[
            "item_id",
            "price",
            "sales_type",
            "sales_rank",
            "categories",
            "title",
            "brand",
        ],
    )

    print("Merging interaction and item data...")
    inter_df_2 = inter_df.join(item_df, on="item_id", how="left")

    print("Sorting by user_id and timestamp...")
    sorted_df = inter_df_2.sort(["user_id", "timestamp"])

    print("Extracting categories and counting frequency...")
    from collections import Counter

    category_counts = Counter()
    for cat_str in item_df["categories"].to_list():
        # Categories are stored as token_seq, e.g. "'Xbox 360', 'Video Games', 'Games'"
        # Split by ", " to get individual categories
        categories = str(cat_str).split(", ")
        for cat in categories:
            # Remove surrounding quotes and whitespace
            cat_clean = cat.strip().strip("'").strip('"').strip()
            if cat_clean:
                category_counts[cat_clean] += 1

    # Get top 100 most popular categories
    top_categories = [cat for cat, _ in category_counts.most_common(100)]
    top_categories_set = set(top_categories)
    print(f"Total unique categories: {len(category_counts)}")
    print(f"Using top 100 categories, rest will be mapped to 'other'")
    print(f"Top 10 categories: {top_categories[:10]}")

    # Create mapping with top 100 + "other" category
    category_to_idx = {cat: i for i, cat in enumerate(top_categories)}
    category_to_idx["other"] = 100
    num_categories = 101
    print(f"Total categories (including 'other'): {num_categories}")

    print("Splitting categories column into list...")
    sorted_df = sorted_df.with_columns(
        pl.col("categories")
        .str.split(", ")
        .map_elements(
            lambda lst: [
                c.strip().strip("'").strip('"').strip() for c in lst if c.strip()
            ],
            return_dtype=pl.List(pl.String),
        )
        .alias("categories_list")
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
        df: Polars DataFrame with columns [user_id, categories_list, rating]
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

    df_with_contribution = df_with_contribution.with_columns(
        [pl.col("user_id").cum_count().over("user_id").alias("interaction_rank")]
    )

    rows = df_with_contribution.to_dicts()

    user_pref_counts = {}
    seen_users = set()
    preference_vectors = []

    for row in tqdm(rows, total=len(rows)):
        user_id = row["user_id"]
        categories = row["categories_list"]
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

        if contributes:
            for cat in categories:
                if cat in category_to_idx:
                    cat_idx = category_to_idx[cat]
                else:
                    cat_idx = category_to_idx["other"]
                user_pref_counts[user_id][cat_idx] += 1

    df = df.with_columns([pl.Series(f"user_pref_vector_{mode}", preference_vectors)])

    return df


if __name__ == "__main__":
    dataset_dir = Path("dataset/Amazon_Video_Games")

    inter_path = dataset_dir / "Amazon_Video_Games.inter"
    item_path = dataset_dir / "Amazon_Video_Games.item"
    output_path = dataset_dir / "Amazon_Video_Games.inter"  # Overwrite original

    prepare_amazon_dataset(inter_path, item_path, output_path)
