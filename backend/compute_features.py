import polars as pl
from config import *
from data_loader import DataLoader
import os

def compute_user_features(lf_events: pl.LazyFrame, df_meta: pl.DataFrame) -> pl.LazyFrame:
    print("Computing user features...")
    
    # 1. Join events with problem metadata to fill missing ratings
    # We cast to LazyFrame because df_meta is eager
    lf_meta = df_meta.lazy().select([
        pl.col(COL_PROBLEM_ID), 
        pl.col("estimated_difficulty")
    ])
    
    lf_enriched = lf_events.join(lf_meta, on=COL_PROBLEM_ID, how="left")
    
    # 2. Sort by user and time
    # Note: Polars requires sorting for rolling
    lf_sorted = lf_enriched.sort([COL_HANDLE, COL_TIMESTAMP])
    
    # Rolling features define the user state (confounders):
    # skill (rating), form (accuracy), difficulty exposure, and activity level
    
    # 3. Define Rolling Expressions
    # We need to compute these effectively.
    # We will use `rolling` over rows (submissions).
    
    # Helper for "is_ok"
    is_ok_expr = (pl.col(COL_VERDICT) == "OK").cast(pl.Int32)
    
    # Difficulty of solved problems (0 if not solved, but we want to ignore it in avg)
    # Strategy: Compute Sum(Diff * OK) / Sum(OK) over window.
    difficulty_val_expr = pl.col("estimated_difficulty").fill_null(0) # handle remaining nulls
    weighted_diff = difficulty_val_expr * is_ok_expr
    
    # Window size from config
    window_size = FUTURE_SUBMISSION_WINDOW
    
    # Rolling features
    feature_exprs = [
        # Rolling Acceptance Rate (bin-safe) - Shifted to avoid leakage
        is_ok_expr.rolling_mean(window_size=window_size)
          .shift(1)
          .over(COL_HANDLE)
          .alias(f"roll_acc_{window_size}"),
        
        # Rolling Count of Solves - Shifted
        is_ok_expr.rolling_sum(window_size=window_size)
          .shift(1)
          .over(COL_HANDLE)
          .alias(f"roll_solve_cnt_{window_size}"),
        
        # Rolling Weighted Diff Sum - Shifted
        weighted_diff.rolling_sum(window_size=window_size)
          .shift(1)
          .over(COL_HANDLE)
          .alias("temp_roll_whook_sum"),
        
        # Time since last event (lag) - Already inherently lagged by logic
        (pl.col(COL_TIMESTAMP) - pl.col(COL_TIMESTAMP).shift(1).over(COL_HANDLE)).alias("time_since_last_sub"),
        
        # Rolling submission velocity - Shifted
        pl.lit(1)
          .rolling_sum(window_size=window_size)
          .shift(1)
          .over(COL_HANDLE)
          .alias(f"roll_sub_cnt_{window_size}")
    ]
    
    lf_features = lf_sorted.with_columns(feature_exprs)
    
    # Post-process rolling average difficulty
    lf_features = lf_features.with_columns([
        (pl.col("temp_roll_whook_sum") / pl.col(f"roll_solve_cnt_{window_size}"))
            .fill_null(0.0)
            .alias(f"roll_ok_diff_{window_size}")
    ]).drop(["temp_roll_whook_sum"])
    
    lf_features = lf_features.with_columns([
        (pl.col(COL_USER_RATING) - pl.col(COL_USER_RATING).shift(1).over(COL_HANDLE))
            .rolling_mean(window_size=window_size)
            .over(COL_HANDLE)
            .alias(f"roll_rating_delta_{window_size}")
    ])
    
    lf_features = lf_features.filter(
        pl.col(f"roll_sub_cnt_{window_size}") >= MIN_SUBMISSIONS_PER_USER
    )
    
    return lf_features

def save_features(lf: pl.LazyFrame):
    print("Collecting and saving features (this may take time)...")
    # We save to partitioned parquet if possible, or single file if fits.
    # 17M rows x 20 cols is manageable in one file (~2GB).
    
    out_path = PROCESSED_DATA_DIR / "user_features.parquet"
    lf.sink_parquet(out_path)
    print(f"Saved user features to {out_path}")

if __name__ == "__main__":
    loader = DataLoader()
    lf_events = loader.clean_and_prepare()
    
    # Load metadata
    meta_path = PROCESSED_DATA_DIR / "problem_metadata.parquet"
    if not meta_path.exists():
        raise FileNotFoundError("Run compute_problem_meta.py first")
        
    df_meta = pl.read_parquet(meta_path)
    
    lf_feats = compute_user_features(lf_events, df_meta)
    save_features(lf_feats)
