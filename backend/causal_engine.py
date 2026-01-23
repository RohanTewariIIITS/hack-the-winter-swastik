import polars as pl
from config import *
import numpy as np
import os

def compute_causal_effects(features_path: Path, min_samples=50):
    print("Loading features...")
    lf = pl.scan_parquet(features_path)
    
    # 1. Compute Outcome: Rating Gain after 20 submissions
    # We used 'rating_at_submission' as current rating.
    # Future rating is the rating at submission i+20.
    
    # Sort just in case
    lf = lf.sort([COL_HANDLE, COL_TIMESTAMP])
    
    lf = lf.with_columns([
        pl.col(COL_USER_RATING).shift(-20).over(COL_HANDLE).alias("future_rating_20")
    ])
    
    lf = lf.with_columns([
        (pl.col("future_rating_20") - pl.col(COL_USER_RATING)).alias("rating_gain_20")
    ])
    
    lf = lf.with_columns([
        (pl.col("rating_gain_20") > 0).cast(pl.Int8).alias("improved_20")
    ])
    
    # Filter out rows where future rating is null (end of history)
    lf = lf.filter(pl.col("rating_gain_20").is_not_null())
    
    # 2. Define State Buckets (Coarsened Matching)
    # Buckets:
    # - Rating: 100 points
    # - Acceptance: 0.1
    # - AvgDiff: 200 points
    
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
    
    # 3. Compute Baseline Gain per Bucket (The "Control")
    # This represents "Average rating growth for a user in this state doing ANY problem"
    # Actually, strictly, the control for "Solving P" is "Solving Not P".
    # Since specific P is rare compared to "Not P", the Global Average is a good approximation of Control.
    
    print("Computing Baseline stats...")
    baseline_stats = (
        lf.group_by(["bin_rating", "bin_acc", "bin_diff"])
        .agg([
            pl.col("rating_gain_20").mean().alias("baseline_gain"),
            pl.col("rating_gain_20").std().alias("baseline_std"),
            pl.col("rating_gain_20").count().alias("baseline_count"),
            pl.col("improved_20").mean().alias("baseline_improve_prob")
        ])
    )
    
    baseline_stats = baseline_stats.filter(
        pl.col("baseline_count") >= min_samples
    )
    
    # 4. Compute Treatment Stats (Per Problem)
    print("Computing Treatment stats...")
    treatment_stats = (
        lf.group_by([COL_PROBLEM_ID, "bin_rating", "bin_acc", "bin_diff"])
        .agg([
            pl.col("rating_gain_20").mean().alias("treated_gain"),
            pl.col("rating_gain_20").std().alias("treated_std"),  # Added STD
            pl.col("rating_gain_20").count().alias("treated_count"),
            pl.col("improved_20").mean().alias("treated_improve_prob")
        ])
    )
    
    # Filter small sample sizes for stability
    treatment_stats = treatment_stats.filter(pl.col("treated_count") >= 10)
    
    # 5. Calculate Effect: (Treated - Baseline) per bucket
    # Join Treatment with Baseline
    effect_df = treatment_stats.join(
        baseline_stats, 
        on=["bin_rating", "bin_acc", "bin_diff"],
        how="inner"
    )
    
    effect_df = effect_df.filter(
        (pl.col("treated_count") >= min_samples) &
        (pl.col("baseline_count") >= min_samples)
    )
    
    effect_df = effect_df.with_columns([
        (pl.col("treated_gain") - pl.col("baseline_gain")).alias("uplift"),
        # Compute Variance of the difference in means for this bucket: s1^2/n1 + s2^2/n2
        ((pl.col("treated_std").pow(2) / pl.col("treated_count")) + 
         (pl.col("baseline_std").pow(2) / pl.col("baseline_count"))).alias("bucket_variance")
    ])
    
    effect_df = effect_df.filter(
        pl.col("uplift").abs() <= 300
    )
    
    effect_df = effect_df.with_columns([
        (pl.col("treated_improve_prob") - pl.col("baseline_improve_prob"))
            .alias("probability_uplift")
    ])
    
    # 6. Aggregate Effects per Problem (ATT) and Compute P-Value
    
    # We can't do complex math like sqrt(sum(weights^2 * variance)) easily in one lazy agg without thinking carefully.
    # ATT = Sum( weight_i * uplift_i )
    # Var(ATT) = Sum( weight_i^2 * bucket_variance_i )
    # where weight_i = treated_count_i / total_treated_count
    
    # Group by problem to compute totals first
    # But lazy execution makes "total_treated_count" (denominator) hard to access inside the group_by before aggregation.
    # We can do it in two steps or using window functions.
    
    # Use window function to get total treated count per problem
    effect_df = effect_df.with_columns([
        pl.col("treated_count").sum().over(COL_PROBLEM_ID).alias("total_treated_samples")
    ])
    
    # Calculate weight per bucket
    effect_df = effect_df.with_columns([
        (pl.col("treated_count") / pl.col("total_treated_samples")).alias("weight")
    ])
    
    final_effects = (
        effect_df.group_by(COL_PROBLEM_ID)
        .agg([
            # ATT = Sum(weight * uplift)
            (pl.col("weight") * pl.col("uplift")).sum().alias("att_score"),
            
            # Var(ATT) = Sum(weight^2 * bucket_variance)
            (pl.col("weight").pow(2) * pl.col("bucket_variance")).sum().alias("att_variance"),
            
            pl.col("total_treated_samples").first().alias("total_treated_samples"),
            pl.col("bin_rating").mean().alias("avg_rating_level"),
            
            ((pl.col("probability_uplift") * pl.col("treated_count")).sum()
             / pl.col("treated_count").sum()).alias("att_probability_uplift")
        ])
    )
    
    # Compute Standard Error, Z-score, P-value using approximations
    # Z = ATT / sqrt(Var(ATT))
    # P = 2 * (1 - CDF(|Z|))
    
    # We'll use 'erf' from scipy or math. But Polars expressions don't have 'erf' directly unless we register a UDF or use apply.
    # UDFs break the lazy plan sometimes or are slower.
    # Let's collect results first (since N problems is small < 30k) then compute P-values in eager mode with numpy/scipy.
    
    return effect_df, final_effects

def run_pipeline():
    features_path = PROCESSED_DATA_DIR / "user_features.parquet"
    print("Running Causal Pipeline...")
    effect_df, final_effects_lazy = compute_causal_effects(features_path)
    
    # Collect to compute P-values
    final_effects = final_effects_lazy.collect()
    
    # Calculate Z-score and P-value eagerly using numpy/scipy
    import scipy.stats as stats
    
    # Standard Error
    final_effects = final_effects.with_columns([
        pl.col("att_variance").sqrt().alias("att_std_err")
    ])
    
    # Z-score
    final_effects = final_effects.with_columns([
        (pl.col("att_score") / pl.col("att_std_err")).alias("z_score")
    ])
    
    # P-value (2-tailed)
    # 2 * (1 - norm.cdf(|z|))
    # We can use map_batches with scipy
    
    z_scores = final_effects["z_score"].to_numpy()
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
    
    final_effects = final_effects.with_columns([
        pl.Series(name="p_value", values=p_values)
    ])
    
    # Filter by Significance
    print(f"Total problems before significance filter: {len(final_effects)}")
    significant_effects = final_effects.filter(pl.col("p_value") < P_VALUE_THRESHOLD)
    print(f"Total problems after significance filter (p < {P_VALUE_THRESHOLD}): {len(significant_effects)}")
    
    # Sort
    significant_effects = significant_effects.sort("att_score", descending=True)
    
    print("Top Causal Problems:")
    print(significant_effects.head(20))
    
    out_path = PROCESSED_DATA_DIR / "causal_att_effects.parquet"
    significant_effects.write_parquet(out_path)
    print(f"Saved aggregated causal effects to {out_path}")
    
    detailed_path = PROCESSED_DATA_DIR / "causal_att_effects_detailed.parquet"
    # effect_df is lazy, collect and write
    
    # NEW: Filter detailed examples for Cohort Analysis (Similiar Users)
    # We want to save actual user examples for the high-impact problems.
    # Since effect_df is aggregated by bucket, we need to go back to the invalid 'lf' or 'treatment_stats' 
    # but we lost user info in aggregation.
    
    # Strategy: Rerun a lightweight scan to get examples for the Top Significant Problems.
    
    sig_problem_ids = significant_effects[COL_PROBLEM_ID].to_list()
    
    # Re-scan to get user examples
    print("Extracting Cohort Examples for UI...")
    lf = pl.scan_parquet(features_path)
    
    # Same pre-processing
    lf = lf.sort([COL_HANDLE, COL_TIMESTAMP])
    lf = lf.with_columns([
        pl.col(COL_USER_RATING).shift(-20).over(COL_HANDLE).alias("future_rating_20")
    ])
    lf = lf.with_columns([
        (pl.col("future_rating_20") - pl.col(COL_USER_RATING)).alias("rating_gain_20")
    ])
    
    # Filter for our top problems only
    cohort_lf = lf.filter(
        (pl.col(COL_PROBLEM_ID).is_in(sig_problem_ids)) &
        (pl.col("rating_gain_20") > 0) # Only positive examples for the UI
    )
    
    # Select needed columns
    cohort_examples = cohort_lf.select([
        COL_HANDLE,
        COL_PROBLEM_ID,
        pl.col(COL_USER_RATING).alias("rating_before"),
        pl.col("future_rating_20").alias("rating_after"),
        pl.col("rating_gain_20").alias("rating_gain")
    ])
    
    # Collect and sample "best" examples per problem
    # Group by problem and take top 5 highest gainers as "Success Stories"
    cohort_examples = cohort_examples.collect()
    
    best_examples = (
        cohort_examples.sort("rating_gain", descending=True)
        .group_by(COL_PROBLEM_ID)
        .head(5)
    )
    
    cohort_path = PROCESSED_DATA_DIR / "cohort_examples.parquet"
    best_examples.write_parquet(cohort_path)
    print(f"Saved {len(best_examples)} cohort examples to {cohort_path}")

    # Save detailed effects (buckets)
    sig_problem_ids_series = significant_effects[COL_PROBLEM_ID]
    detailed_filtered = effect_df.filter(pl.col(COL_PROBLEM_ID).is_in(sig_problem_ids_series)).collect()
    
    detailed_filtered.write_parquet(detailed_path)
    print(f"Saved detailed causal effects to {detailed_path}")

if __name__ == "__main__":
    run_pipeline()
