import polars as pl
from config import *
import httpx
import asyncio

class RecommenderService:
    def __init__(self):
        self.detailed_effects = None
        self.meta_df = None
        self.loaded = False

    async def lookup_user_profile(self, handle: str):
        """
        Look up user stats locally from user_features.parquet
        instead of calling Codeforces API.
        """
        features_path = PROCESSED_DATA_DIR / "user_features.parquet"
        
        if not features_path.exists():
            print("Local features file not found.")
            return None
            
        try:
            # fast scan for specific handle
            # sort by time desc to get latest state
            user_data = (
                pl.scan_parquet(features_path)
                .filter(pl.col(COL_HANDLE) == handle)
                .sort(COL_TIMESTAMP, descending=True)
                .select([
                    COL_USER_RATING,
                    "roll_acc_20",     # Recent accuracy
                    "roll_ok_diff_20"  # Recent avg difficulty
                ])
                .head(1)
                .collect()
            )
            
            if user_data.is_empty():
                print(f"User {handle} not found in local dataset.")
                return None
                
            row = user_data.row(0)
            current_rating = row[0]
            recent_acc = row[1] if row[1] is not None else 0.5
            recent_diff = row[2] if row[2] is not None else current_rating
            
            return {
                "handle": handle,
                "current_rating": int(current_rating),
                "rank": "local_user", # We don't store rank locally
                "recent_accuracy": round(recent_acc, 2),
                "recent_avg_difficulty": round(recent_diff, 1)
            }
            
        except Exception as e:
            print(f"Local Lookup Error: {e}")
            return None

    def load_data(self):
        print("Loading recommender data (V2)...")
        detailed_path = PROCESSED_DATA_DIR / "causal_att_effects.parquet" 
        meta_path = PROCESSED_DATA_DIR / "problem_metadata.parquet"
        survival_path = PROCESSED_DATA_DIR / "survival_effects.parquet"
        sequence_path = PROCESSED_DATA_DIR / "sequence_patterns.parquet"
        
        # V2 New Paths
        cohort_path = PROCESSED_DATA_DIR / "cohort_examples.parquet"
        graph_path = PROCESSED_DATA_DIR / "problem_graph.parquet"
        
        if not detailed_path.exists():
            raise FileNotFoundError("Run causal_engine.py first")
            
        # Load tables
        self.detailed_effects = pl.read_parquet(detailed_path).with_columns(pl.col(COL_PROBLEM_ID).cast(pl.String))
        self.meta_df = pl.read_parquet(meta_path).with_columns(pl.col(COL_PROBLEM_ID).cast(pl.String))
        
        if survival_path.exists():
            print("Loading survival analysis effects...")
            survival_df = pl.read_parquet(survival_path).with_columns(pl.col(COL_PROBLEM_ID).cast(pl.String))
            self.detailed_effects = self.detailed_effects.join(
                survival_df.select([COL_PROBLEM_ID, "median_time_to_improve", "hazard_ratio"]),
                on=COL_PROBLEM_ID, how="left"
            )
        else:
            self.detailed_effects = self.detailed_effects.with_columns([
                pl.lit(None).cast(pl.Float64).alias("median_time_to_improve"),
                pl.lit(1.0).cast(pl.Float64).alias("hazard_ratio")
            ])
            
        # Sequence patterns file has different schema (pattern, support, confidence)
        # Skip the join and just add a default column
        self.detailed_effects = self.detailed_effects.with_columns([pl.lit(0.0).cast(pl.Float64).alias("sequence_confidence")])
            

        if cohort_path.exists():
            print("Loading cohort examples...")
            # Cohort data uses COL_PROBLEM_ID from seed
            self.cohort_df = pl.read_parquet(cohort_path)
        else:
            self.cohort_df = None
            
        if graph_path.exists():
            print("Loading problem graph...")
            # Graph uses 'source' and 'target' columns from seed
            self.graph_df = pl.read_parquet(graph_path)
        else:
            self.graph_df = None

        
        # Join Metadata
        self.detailed_effects = self.detailed_effects.join(
            self.meta_df.select([COL_PROBLEM_ID, "estimated_difficulty", "acceptance_rate"]),
            on=COL_PROBLEM_ID, how="left"
        )
        
        # Fill Nulls
        self.detailed_effects = self.detailed_effects.with_columns([
             pl.col("sequence_confidence").fill_null(0.0),
             pl.col("hazard_ratio").fill_null(1.0)
        ])
        
        self.loaded = True
        print(f"Loaded {len(self.detailed_effects)} significant causal rules (V2).")

    def get_global_insights(self, top_k=20):
        if not self.loaded: self.load_data()
        
        # Return top problems sorted by ATT score (Impact)
        top = self.detailed_effects.sort("att_score", descending=True).head(top_k)
        # Rename id column to 'problem_id' for frontend compatibility
        top = top.rename({COL_PROBLEM_ID: "problem_id"})
        return top.to_dicts()

        
    def get_problem_details(self, problem_id: str):
        if not self.loaded: self.load_data()
        
        # Get Stats
        stats = self.detailed_effects.filter(pl.col(COL_PROBLEM_ID) == problem_id)
        if stats.is_empty():
            return None
        
        stats_dict = stats.to_dicts()[0]
        
        # Get Cohorts (Similar Users)
        cohorts = []
        if self.cohort_df is not None:
             cols = self.cohort_df.filter(pl.col(COL_PROBLEM_ID) == problem_id)
             cohorts = cols.to_dicts()
             
        # Get Next Steps (Graph)
        next_steps = []
        if self.graph_df is not None:
            edges = self.graph_df.filter(pl.col("source_problem") == problem_id)
            next_steps = edges.sort("transition_probability", descending=True).head(5).to_dicts()
            
        return {
            "stats": stats_dict,
            "similar_users": cohorts,
            "next_steps": next_steps
        }

    def recommend(self, current_rating: float, recent_acc: float, recent_diff_avg: float, top_k: int = 5):
        if not self.loaded:
            self.load_data()
            
        # 1. Bucketize User Input & Filter Candidates
        candidates = self.detailed_effects.filter(
            (pl.col("estimated_difficulty") >= current_rating - 200) &
            (pl.col("estimated_difficulty") <= current_rating + 400)
        )
        
        if len(candidates) == 0:
            candidates = self.detailed_effects
        
        # 2. Multi-Objective Ranking
        candidates = candidates.with_columns([
            (
                pl.col("att_score") * 
                pl.col("att_probability_uplift") * 
                pl.col("hazard_ratio") * 
                (1 + pl.col("sequence_confidence"))
            ).alias("final_score")
        ])
        
        recommendations = candidates.sort("final_score", descending=True).head(top_k)
        
        results = []
        for row in recommendations.iter_rows(named=True):
            problem_id = row[COL_PROBLEM_ID]
            # ... existing logic ...
            explanation = (
                f"Highly recommended (Score: {row['final_score']:.1f}). "
                f"Causal Uplift: +{row['att_score']:.1f} points (p={row['p_value']:.4f}). "
                f"Solvers improve {row['hazard_ratio']:.1f}x faster."
            )
            
            results.append({
                "problem_id": problem_id,
                "uplift": float(row['att_score']),
                "probability_uplift": float(row['att_probability_uplift']),
                "p_value": float(row['p_value']),
                "hazard_ratio": float(row['hazard_ratio']),
                "median_time_to_improve": float(row['median_time_to_improve']) if row['median_time_to_improve'] is not None else -1,
                "estimated_difficulty": row["estimated_difficulty"],
                "explanation": explanation,
                "sample_size": row["total_treated_samples"]
            })
            
        return results

# Singleton instance
rec_service = RecommenderService()
