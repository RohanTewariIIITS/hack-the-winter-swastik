

import polars as pl
from config import *
from pathlib import Path

"""
survival_engine.py

Purpose:
Estimate HOW FAST rating improvement happens after solving a problem.
Implements survival-style analysis (time-to-improvement) at problem level.

Output:
data/processed/survival_effects.parquet
"""

def compute_survival_effects(features_path: Path, improvement_delta: int = 50):
    """
    For each (user, problem) pair:
    - Start time t0 = when problem is solved
    - Event occurs if rating increases by `improvement_delta`
    - Time measured in number of submissions
    """

    lf = pl.scan_parquet(features_path)

    # Ensure proper ordering
    lf = lf.sort([COL_HANDLE, COL_TIMESTAMP])

    # Identify successful solves (treatment points)
    lf = lf.with_columns([
        (pl.col(COL_VERDICT) == "OK").cast(pl.Int8).alias("is_solve")
    ])

    # Rating at solve time
    lf = lf.with_columns([
        pl.when(pl.col("is_solve") == 1)
          .then(pl.col(COL_USER_RATING))
          .otherwise(None)
          .alias("rating_at_solve")
    ])

    # Forward fill rating_at_solve until next solve
    lf = lf.with_columns([
        pl.col("rating_at_solve")
          .forward_fill()
          .over(COL_HANDLE)
    ])

    # Submission index per user (acts as time axis)
    lf = lf.with_columns([
        pl.int_range(0, pl.count()).over(COL_HANDLE).alias("submission_idx")
    ])

    # Identify improvement event
    lf = lf.with_columns([
        (
            pl.col(COL_USER_RATING)
            >= pl.col("rating_at_solve") + improvement_delta
        ).cast(pl.Int8).alias("improved_event")
    ])

    # Keep only rows AFTER a solve happened
    lf = lf.filter(pl.col("rating_at_solve").is_not_null())

    # First improvement time after solve
    event_times = (
        lf.filter(pl.col("improved_event") == 1)
          .group_by([COL_HANDLE, COL_PROBLEM_ID])
          .agg([
              pl.col("submission_idx").min().alias("event_time")
          ])
    )

    # Solve time index
    solve_times = (
        lf.filter(pl.col("is_solve") == 1)
          .group_by([COL_HANDLE, COL_PROBLEM_ID])
          .agg([
              pl.col("submission_idx").min().alias("solve_time")
          ])
    )

    # Join event & solve times
    survival_df = solve_times.join(
        event_times,
        on=[COL_HANDLE, COL_PROBLEM_ID],
        how="left"
    ).with_columns([
        (pl.col("event_time") - pl.col("solve_time")).alias("time_to_event"),
        pl.col("event_time").is_not_null().cast(pl.Int8).alias("event_observed")
    ])

    # Remove invalid / negative times
    survival_df = survival_df.filter(
        (pl.col("time_to_event") >= 0) | (pl.col("event_observed") == 0)
    )

    # Aggregate per problem
    problem_survival = (
        survival_df.group_by(COL_PROBLEM_ID)
        .agg([
            pl.col("event_observed").mean().alias("event_rate"),
            pl.col("time_to_event").median().alias("median_time_to_improve"),
            pl.count().alias("sample_size")
        ])
        .filter(pl.col("sample_size") >= MIN_TREATED_SAMPLES)
    )

    # Convert event rate to hazard-style signal
    problem_survival = problem_survival.with_columns([
        (pl.col("event_rate") / pl.col("event_rate").mean())
            .fill_null(1.0)
            .alias("hazard_ratio")
    ])

    return problem_survival


def run_survival_pipeline():
    print("Running Survival Analysis Engine...")

    features_path = PROCESSED_DATA_DIR / "user_features.parquet"
    if not features_path.exists():
        raise FileNotFoundError("user_features.parquet not found. Run compute_features.py first.")

    survival_df = compute_survival_effects(features_path)
    
    # Collect LazyFrame to DataFrame before writing
    survival_df = survival_df.collect()

    out_path = PROCESSED_DATA_DIR / "survival_effects.parquet"
    survival_df.write_parquet(out_path)

    print(f"Saved survival effects to {out_path}")
    print(survival_df.sort("hazard_ratio", descending=True).head(10))


if __name__ == "__main__":
    run_survival_pipeline()