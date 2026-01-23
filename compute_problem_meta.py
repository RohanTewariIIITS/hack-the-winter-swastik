import polars as pl
from config import *
from data_loader import DataLoader
import os

def compute_problem_metadata(lf: pl.LazyFrame) -> pl.DataFrame:
    print("Computing problem metadata...")
    
    # Filter for successful submissions to estimate difficulty
    # If problem_rating is in the dataset, use it. If null, impute.
    # Group by problem_id
    
    # 1. Base stats including null problem_ratings
    problem_stats = (
        lf.group_by(COL_PROBLEM_ID)
        .agg([
            pl.col(COL_PROBLEM_RATING).mean().alias("avg_provided_rating"),
            pl.col(COL_USER_RATING).filter(pl.col(COL_VERDICT) == "OK").mean().alias("avg_solver_rating"),
            pl.col(COL_VERDICT).count().alias("total_submissions"),
            (pl.col(COL_VERDICT) == "OK").sum().alias("ok_count")
        ])
    )
    
    # Execute to get DataFrame (problems are fewer than submissions)
    df_problems = problem_stats.collect()

    # 1. Add minimum submission support and flags
    df_problems = df_problems.with_columns([
        (pl.col("total_submissions") >= 50).alias("has_sufficient_support")
    ])

    # 2. Robust difficulty estimation (shrinkage)
    # Shrinkage-based difficulty estimation
    alpha = 0.7  # trust provided ratings more when available
    df_problems = df_problems.with_columns([
        pl.when(pl.col("avg_provided_rating").is_not_null())
          .then(
              alpha * pl.col("avg_provided_rating") +
              (1 - alpha) * pl.col("avg_solver_rating")
          )
          .otherwise(pl.col("avg_solver_rating"))
          .alias("estimated_difficulty"),

        (pl.col("ok_count") / pl.col("total_submissions"))
            .fill_nan(0.0)
            .alias("acceptance_rate")
    ])

    # 3. Add acceptance-rate stability flags
    df_problems = df_problems.with_columns([
        (pl.col("acceptance_rate") >= 0.05).alias("stable_acceptance")
    ])

    # 4. Cap extreme difficulty values
    df_problems = df_problems.with_columns([
        pl.col("estimated_difficulty")
          .clip(MIN_RATING, MAX_RATING)
          .alias("estimated_difficulty")
    ])

    print(f"Computed metadata for {len(df_problems)} problems.")
    return df_problems

def save_problem_metadata(df: pl.DataFrame):
    out_path = PROCESSED_DATA_DIR / "problem_metadata.parquet"
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)
    print(f"Saved problem metadata to {out_path}")

if __name__ == "__main__":
    loader = DataLoader()
    lf = loader.clean_and_prepare()
    df_meta = compute_problem_metadata(lf)
    # 5. Persist additional metadata for downstream causal use
    expected_cols = {
        "estimated_difficulty",
        "acceptance_rate",
        "has_sufficient_support",
        "stable_acceptance"
    }
    missing = expected_cols - set(df_meta.columns)
    if missing:
        raise ValueError(f"Missing required problem metadata columns: {missing}")
    save_problem_metadata(df_meta)
    # 6. Improve console diagnostics
    print(df_meta.select([
        pl.count().alias("total_problems"),
        pl.col("has_sufficient_support").sum().alias("supported_problems"),
        pl.col("stable_acceptance").sum().alias("stable_acceptance_problems")
    ]))
    print(df_meta.head())
