import polars as pl
from config import *

def seed_data():
    print("Seeding comprehensive demo data for Codeforces-like platform...")
    
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # ==========================================================================
    # 1. USER PROFILES - Multiple users for demo switching
    # ==========================================================================
    print("Seeding multi-user profiles...")
    
    users = [
        # Legendary Grandmasters
        {"handle": "tourist", "rating": 3995, "accuracy": 0.92, "avg_diff": 2800},
        {"handle": "jiangly", "rating": 3608, "accuracy": 0.89, "avg_diff": 2600},
        {"handle": "ecnerwala", "rating": 3400, "accuracy": 0.87, "avg_diff": 2500},
        {"handle": "rainboy", "rating": 3100, "accuracy": 0.85, "avg_diff": 2400},
        # Candidate Masters / Experts (Typical hackathon demo range)
        {"handle": "candidate_master_1", "rating": 1920, "accuracy": 0.72, "avg_diff": 1600},
        {"handle": "expert_user_1", "rating": 1650, "accuracy": 0.68, "avg_diff": 1400},
        {"handle": "specialist_user", "rating": 1450, "accuracy": 0.62, "avg_diff": 1200},
        {"handle": "pupil_user", "rating": 1250, "accuracy": 0.55, "avg_diff": 1000},
    ]
    
    user_features = pl.DataFrame({
        COL_HANDLE: [u["handle"] for u in users],
        COL_TIMESTAMP: [1700000000] * len(users),
        COL_USER_RATING: [float(u["rating"]) for u in users],
        "roll_acc_20": [u["accuracy"] for u in users],
        "roll_ok_diff_20": [float(u["avg_diff"]) for u in users]
    })
    user_features.write_parquet(PROCESSED_DATA_DIR / "user_features.parquet")
    
    # ==========================================================================
    # 2. PROBLEMSET - Rich problem data with titles and tags
    # ==========================================================================
    print("Seeding problemset with titles and tags...")
    
    problems = [
        # High-impact problems (1400-1600 range)
        {"id": "1462F", "title": "The Treasure of The Segments", "diff": 1500, "tags": "Greedy, Sortings, Two Pointers", "att": 61.2, "prob": 0.22, "p": 0.0001, "samples": 450, "time": 150, "hr": 1.8},
        {"id": "1623C", "title": "Balanced Stone Heaps", "diff": 1600, "tags": "Binary Search, Greedy", "att": 58.4, "prob": 0.20, "p": 0.0002, "samples": 300, "time": 200, "hr": 1.6},
        {"id": "1552E", "title": "Colors and Intervals", "diff": 1600, "tags": "Constructive, Greedy, Sortings", "att": 42.5, "prob": 0.18, "p": 0.001, "samples": 240, "time": 250, "hr": 1.5},
        {"id": "1709E", "title": "XOR Tree", "diff": 1700, "tags": "DFS, Trees, Bitmasks", "att": 55.0, "prob": 0.19, "p": 0.001, "samples": 280, "time": 220, "hr": 1.4},
        {"id": "1705C", "title": "Mark and His Unfinished Essay", "diff": 1700, "tags": "Binary Search, Implementation", "att": 37.1, "prob": 0.15, "p": 0.005, "samples": 180, "time": 300, "hr": 1.3},
        # Medium-impact problems (1700-1900 range)
        {"id": "1800E", "title": "Unforgivable Curse", "diff": 1800, "tags": "Constructive, Strings", "att": 48.0, "prob": 0.17, "p": 0.002, "samples": 320, "time": 180, "hr": 1.5},
        {"id": "1829G", "title": "Hits Different", "diff": 1800, "tags": "DP, Math, Implementation", "att": 45.2, "prob": 0.16, "p": 0.003, "samples": 290, "time": 210, "hr": 1.4},
        {"id": "1850H", "title": "The Third Letter", "diff": 1900, "tags": "DSU, Graphs, Implementation", "att": 52.3, "prob": 0.18, "p": 0.002, "samples": 260, "time": 190, "hr": 1.5},
        # Lower difficulty (1200-1400 range)
        {"id": "1914E", "title": "Game with Marbles", "diff": 1400, "tags": "Greedy, Sortings, Games", "att": 35.0, "prob": 0.14, "p": 0.008, "samples": 520, "time": 120, "hr": 1.3},
        {"id": "1899F", "title": "Alex's whims", "diff": 1400, "tags": "Constructive, Trees", "att": 32.5, "prob": 0.13, "p": 0.01, "samples": 480, "time": 140, "hr": 1.2},
        {"id": "1881E", "title": "Block Sequence", "diff": 1300, "tags": "DP, Two Pointers", "att": 28.0, "prob": 0.12, "p": 0.01, "samples": 600, "time": 100, "hr": 1.2},
        {"id": "1878E", "title": "Iva & Pav", "diff": 1200, "tags": "Binary Search, Bitmasks", "att": 25.5, "prob": 0.11, "p": 0.01, "samples": 700, "time": 90, "hr": 1.1},
    ]
    
    # Causal Effects (for recommendations)
    causal_effects = pl.DataFrame({
        COL_PROBLEM_ID: [p["id"] for p in problems],
        "att_score": [p["att"] for p in problems],
        "att_probability_uplift": [p["prob"] for p in problems],
        "p_value": [p["p"] for p in problems],
        "total_treated_samples": [p["samples"] for p in problems],
        "estimated_difficulty": [float(p["diff"]) for p in problems]
    })
    causal_effects.write_parquet(PROCESSED_DATA_DIR / "causal_att_effects.parquet")
    
    # Problem Metadata (titles, tags, acceptance rates)
    meta = pl.DataFrame({
        COL_PROBLEM_ID: [p["id"] for p in problems],
        "title": [p["title"] for p in problems],
        "tags": [p["tags"] for p in problems],
        "estimated_difficulty": [float(p["diff"]) for p in problems],
        "acceptance_rate": [0.45, 0.42, 0.45, 0.39, 0.38, 0.35, 0.36, 0.33, 0.52, 0.50, 0.55, 0.58]
    })
    meta.write_parquet(PROCESSED_DATA_DIR / "problem_metadata.parquet")
    
    # ==========================================================================
    # 3. SURVIVAL & SEQUENCE DATA
    # ==========================================================================
    print("Seeding survival analysis data...")
    survival = pl.DataFrame({
        COL_PROBLEM_ID: [p["id"] for p in problems],
        "median_time_to_improve": [float(p["time"]) for p in problems],
        "hazard_ratio": [p["hr"] for p in problems]
    })
    survival.write_parquet(PROCESSED_DATA_DIR / "survival_effects.parquet")
    
    # Sequence patterns (placeholder)
    sequence = pl.DataFrame({
        "pattern": ["1462F->1623C", "1552E->1709E", "1914E->1462F"],
        "support": [0.15, 0.12, 0.18],
        "confidence": [0.65, 0.58, 0.72]
    })
    sequence.write_parquet(PROCESSED_DATA_DIR / "sequence_patterns.parquet")
    
    # ==========================================================================
    # 4. COHORT EXAMPLES (Similar Users)
    # ==========================================================================
    print("Seeding cohort examples (success stories)...")
    cohorts = pl.DataFrame({
        COL_HANDLE: [
            "user_123", "algo_guy", "cf_runner", "coder_x", "rank_up",
            "cp_master", "rising_star", "grind_mode", "practice_daily", "improvement_arc"
        ],
        COL_PROBLEM_ID: [
            "1462F", "1462F", "1462F", "1623C", "1623C",
            "1552E", "1709E", "1800E", "1914E", "1881E"
        ],
        "rating_before": [1890, 1910, 1865, 1750, 1780, 1820, 1680, 1720, 1520, 1380],
        "rating_after": [1951, 1968, 1920, 1808, 1842, 1875, 1742, 1778, 1572, 1428],
        "rating_gain": [61, 58, 55, 58, 62, 55, 62, 58, 52, 48]
    })
    cohorts.write_parquet(PROCESSED_DATA_DIR / "cohort_examples.parquet")
    
    # Problem graph (placeholder)
    graph = pl.DataFrame({
        "source": ["1462F", "1623C", "1552E"],
        "target": ["1623C", "1709E", "1800E"],
        "weight": [0.35, 0.28, 0.22]
    })
    graph.write_parquet(PROCESSED_DATA_DIR / "problem_graph.parquet")

    print("✅ Demo data seeded successfully!")
    print(f"   Users: {len(users)}")
    print(f"   Problems: {len(problems)}")

if __name__ == "__main__":
    seed_data()
