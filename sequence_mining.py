

import polars as pl
from config import *
from pathlib import Path

"""
sequence_mining.py

Purpose:
Discover problem-solving SEQUENCES that consistently precede rating improvement.
This captures *learning paths*, not isolated problems.

Output:
data/processed/sequence_patterns.parquet
"""

def compute_sequence_patterns(
    features_path: Path,
    improvement_delta: int = 100,
    max_seq_len: int = 3,
    min_support: int = 30
):
    """
    Builds sequences of solved problems per user and detects
    which problems frequently appear RIGHT BEFORE rating improvement.

    This is a scalable approximation of sequential pattern mining.
    """

    lf = pl.scan_parquet(features_path)

    # Ensure chronological order
    lf = lf.sort([COL_HANDLE, COL_TIMESTAMP])

    # Identify successful solves
    lf = lf.with_columns([
        (pl.col(COL_VERDICT) == "OK").cast(pl.Int8).alias("is_solve")
    ])

    # Rating change between consecutive submissions
    lf = lf.with_columns([
        (pl.col(COL_USER_RATING) - pl.col(COL_USER_RATING).shift(1).over(COL_HANDLE))
            .alias("rating_delta")
    ])

    # Rolling cumulative improvement marker
    lf = lf.with_columns([
        (
            pl.col("rating_delta")
            .rolling_sum(window_size=PAST_SUBMISSION_WINDOW)
            .over(COL_HANDLE)
            >= improvement_delta
        ).cast(pl.Int8).alias("improved_event")
    ])

    # Keep only rows where an improvement just happened
    improved_points = lf.filter(pl.col("improved_event") == 1)

    # For each improvement point, fetch last K solved problems
    seq_rows = []

    df = improved_points.collect()
    full_df = lf.collect()

    for row in df.iter_rows(named=True):
        user = row[COL_HANDLE]
        ts = row[COL_TIMESTAMP]

        user_history = full_df.filter(
            (pl.col(COL_HANDLE) == user) &
            (pl.col(COL_TIMESTAMP) < ts) &
            (pl.col("is_solve") == 1)
        ).sort(COL_TIMESTAMP, descending=True)

        problems = user_history.select(COL_PROBLEM_ID).head(max_seq_len)[COL_PROBLEM_ID].to_list()

        for p in problems:
            seq_rows.append({
                COL_PROBLEM_ID: p,
                "sequence_confidence": 1
            })

    if not seq_rows:
        return pl.DataFrame(
            [],
            schema={
                COL_PROBLEM_ID: pl.String,
                "sequence_confidence": pl.Float64,
                "support": pl.UInt32
            }
        )

    seq_df = pl.DataFrame(seq_rows)

    # Aggregate sequence confidence
    seq_patterns = (
        seq_df.group_by(COL_PROBLEM_ID)
        .agg([
            pl.count().alias("support")
        ])
        .filter(pl.col("support") >= min_support)
    )

    # Normalize confidence
    seq_patterns = seq_patterns.with_columns([
        (pl.col("support") / pl.col("support").max())
            .alias("sequence_confidence")
    ])

    return seq_patterns


def run_sequence_pipeline():
    print("Running Sequential Pattern Mining Engine...")

    features_path = PROCESSED_DATA_DIR / "user_features.parquet"
    if not features_path.exists():
        raise FileNotFoundError("user_features.parquet not found. Run compute_features.py first.")

    seq_df = compute_sequence_patterns(features_path)

    out_path = PROCESSED_DATA_DIR / "sequence_patterns.parquet"
    seq_df.write_parquet(out_path)

    print(f"Saved sequence patterns to {out_path}")
    print(seq_df.sort("sequence_confidence", descending=True).head(10))


if __name__ == "__main__":
    run_sequence_pipeline()