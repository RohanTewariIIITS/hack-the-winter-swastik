import polars as pl
from config import *
import numpy as np

def run_placebo_test(features_path: Path):
    print("Running Placebo Test (Reverse Causality)...")
    lf = pl.scan_parquet(features_path)
    
    # 1. Compute Placebo Outcome: Rating Gain *Before* Solve
    # Gain from t-20 to t.
    # Note: We use t-20 rating vs current rating.
    
    lf = lf.sort([COL_HANDLE, COL_TIMESTAMP])
    
    lf = lf.with_columns([
        pl.col(COL_USER_RATING).shift(20).over(COL_HANDLE).alias("past_rating_20")
    ])
    
    # Placebo Gain: Current - Past
    lf = lf.with_columns([
        (pl.col(COL_USER_RATING) - pl.col("past_rating_20")).alias("placebo_gain_20")
    ])
    
    lf = lf.filter(pl.col("placebo_gain_20").is_not_null())
    
    lf = lf.with_columns([
        (pl.col("placebo_gain_20") > 0).cast(pl.Int8).alias("placebo_improved_20")
    ])
    
    # 2. Coarsened Matching (Same State Definition)
    lf = lf.with_columns([
        (pl.col(COL_USER_RATING) / 100).round(0).cast(pl.Int32).alias("bin_rating"),
        (pl.col("roll_acc_20") * 10).round(0).cast(pl.Int32).alias("bin_acc"),
        (
            pl.when(pl.col("roll_ok_diff_20").is_infinite())
            .then(0.0)
            .otherwise(pl.col("roll_ok_diff_20").fill_nan(0.0)) 
            / 200
        ).round(0).cast(pl.Int32).alias("bin_diff")
    ])
    
    # 3. Compute Baseline Placebo
    baseline_placebo = (
        lf.group_by(["bin_rating", "bin_acc", "bin_diff"])
        .agg([
            pl.col("placebo_gain_20").mean().alias("baseline_placebo_gain"),
            pl.col("placebo_improved_20").mean().alias("baseline_placebo_improve_prob")
        ])
    )
    
    baseline_placebo = baseline_placebo.with_columns([
        pl.count().over(["bin_rating", "bin_acc", "bin_diff"]).alias("baseline_count")
    ]).filter(
        pl.col("baseline_count") >= 20
    )
    
    # 4. Compute Treated Placebo
    treated_placebo = (
        lf.group_by([COL_PROBLEM_ID, "bin_rating", "bin_acc", "bin_diff"])
        .agg([
            pl.col("placebo_gain_20").mean().alias("treated_placebo_gain"),
            pl.col("placebo_gain_20").count().alias("treated_count"),
            pl.col("placebo_improved_20").mean().alias("treated_placebo_improve_prob")
        ])
    ).filter(pl.col("treated_count") >= 20)
    
    # 5. Join and Compute Bias
    bias_df = treated_placebo.join(
        baseline_placebo,
        on=["bin_rating", "bin_acc", "bin_diff"],
        how="inner"
    )
    
    bias_df = bias_df.with_columns([
        (pl.col("treated_placebo_gain") - pl.col("baseline_placebo_gain")).alias("selection_bias")
    ])
    
    bias_df = bias_df.filter(
        pl.col("selection_bias").abs() <= 300
    )
    
    bias_df = bias_df.with_columns([
        (pl.col("treated_placebo_improve_prob") - pl.col("baseline_placebo_improve_prob"))
            .alias("probability_selection_bias")
    ])
    
    # 6. Aggregate Bias per Problem
    aggregated_bias = (
        bias_df.group_by(COL_PROBLEM_ID)
        .agg([
            ((pl.col("selection_bias") * pl.col("treated_count")).sum() / pl.col("treated_count").sum()).alias("bias_score"),
            ((pl.col("probability_selection_bias") * pl.col("treated_count")).sum()
             / pl.col("treated_count").sum()).alias("probability_bias_score"),
            pl.col("treated_count").sum().alias("total_samples")
        ])
    ).sort("bias_score", descending=True)
    
    print("\nTop Problems with Positive Selection Bias (Users improve BEFORE solving them):")
    res = aggregated_bias.collect()
    print(res.head(10))
    
    mean_bias = res["bias_score"].mean()
    if mean_bias is not None:
        print(f"\nAverage System Bias: {mean_bias:.2f}")
    else:
        print("\nAverage System Bias: N/A (No bias data found)")
    
    print("Problems with strongest reverse causality detected (selection bias):")
    
    out_path = PROCESSED_DATA_DIR / "placebo_selection_bias.parquet"
    res.write_parquet(out_path)
    print(f"Saved bias report to {out_path}")

if __name__ == "__main__":
    run_placebo_test(PROCESSED_DATA_DIR / "user_features.parquet")
