
import polars as pl
from config import *
from pathlib import Path

"""
graph_engine.py

Purpose:
Construct a Directed Graph of Problem Transitions.
Nodes: Problems
Edges: User solved B within GRAPH_TRANSITION_WINDOW submissions after A.

This models "Behavioral Learning Pathways": what users actually solve next.
It allows us to constrain recommendations to "plausible" next steps.

Output:
data/processed/problem_graph.parquet
"""

def compute_problem_graph(features_path: Path):
    print("Building Problem Transition Graph...")
    lf = pl.scan_parquet(features_path)
    
    # 1. Sort by User and Time
    lf = lf.sort([COL_HANDLE, COL_TIMESTAMP])
    
    # 2. Filter for Solved Problems Only
    solved_lf = lf.filter(pl.col(COL_VERDICT) == "OK").select([
        COL_HANDLE, 
        COL_PROBLEM_ID, 
        COL_TIMESTAMP
    ])
    
    # Add index (acts as time)
    solved_lf = solved_lf.with_columns([
        pl.int_range(0, pl.count()).over(COL_HANDLE).alias("solve_idx")
    ])

    # 3. Create Source (A) -> Target (B) Pairs
    # We look efficiently for next k solved problems.
    # Self-join approach on user, where rank_B - rank_A <= window
    
    # Since window is small (e.g. 5), can use shift columns.
    
    edge_dfs = []
    
    for gap in range(1, GRAPH_TRANSITION_WINDOW + 1):
        # Shift to get "Next Problem" at gap `gap`
        next_problems = solved_lf.with_columns([
            pl.col(COL_PROBLEM_ID).shift(-gap).over(COL_HANDLE).alias("target_problem"),
            (pl.col("solve_idx").shift(-gap).over(COL_HANDLE) - pl.col("solve_idx")).alias("idx_diff")
        ])
        
        # Filter valid transitions
        # idx_diff should be exactly 'gap' if consecutive solves (ignoring Failed submissions in between)
        # But here 'solved_lf' ONLY has solves. So idx_diff is exactly gap.
        # Just filter out nulls (end of history).
        
        edges = next_problems.filter(pl.col("target_problem").is_not_null())
        
        edge_dfs.append(edges.select([
            pl.col(COL_PROBLEM_ID).alias("source_problem"),
            pl.col("target_problem"),
            pl.lit(gap).alias("gap_size")
        ]))
        
    # Combine all gaps
    all_edges = pl.concat(edge_dfs)
    
    # 4. Aggregate Edges
    graph_edges = (
        all_edges.group_by(["source_problem", "target_problem"])
        .agg([
            pl.count().alias("transition_count"),
            pl.col("gap_size").mean().alias("avg_gap")
        ])
    )
    
    # 5. Filter for Significant Pathways
    graph_edges = graph_edges.filter(
        pl.col("transition_count") >= MIN_TRANSITION_COUNT
    )
    
    # 6. Normalize Edge Weights (Transition Probability)
    graph_edges = graph_edges.with_columns([
        (pl.col("transition_count") / pl.col("transition_count").sum().over("source_problem"))
            .alias("transition_probability")
    ])
    
    return graph_edges

def run_graph_pipeline():
    features_path = PROCESSED_DATA_DIR / "user_features.parquet"
    if not features_path.exists():
        raise FileNotFoundError("user_features.parquet not found.")
        
    graph_df = compute_problem_graph(features_path)
    
    # Collect
    res = graph_df.collect()
    
    print(f"Discovered {len(res)} graph edges.")
    print("Top Transitions:")
    print(res.sort("transition_count", descending=True).head(10))
    
    out_path = PROCESSED_DATA_DIR / "problem_graph.parquet"
    res.write_parquet(out_path)
    print(f"Saved problem graph to {out_path}")

if __name__ == "__main__":
    run_graph_pipeline()
