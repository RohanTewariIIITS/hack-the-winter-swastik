from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Dataset
DATASET_NAME = "denkCF/UsersCodeforcesSubmissionsEnd2024"

# Schema Constants
COL_HANDLE = "handle"
COL_USER_RATING = "rating_at_submission"
COL_PROBLEM_RATING = "problem_rating"
COL_PROBLEM_ID = "id_of_submission_task"
COL_VERDICT = "verdict"
COL_TIMESTAMP = "time"

# Processing

MIN_SUBMISSIONS_PER_USER = 10
MIN_RATING = 800
MAX_RATING = 4000

# Temporal Windows (Causal)
FUTURE_SUBMISSION_WINDOW = 20   # submissions after solving a problem
PAST_SUBMISSION_WINDOW = 20     # submissions before solving (placebo)

# Coarsened Exact Matching (CEM) Bin Sizes
RATING_BIN_SIZE = 100           # CF rating buckets
ACCURACY_BIN_SIZE = 0.1         # rolling accuracy buckets
DIFFICULTY_BIN_SIZE = 200       # difficulty buckets

# Causal Stability Thresholds
MIN_BASELINE_SAMPLES = 20       # minimum control samples per state
MIN_TREATED_SAMPLES = 20        # minimum treated samples per state
MAX_ALLOWED_UPLIFT = 300        # rating points (sanity cap)

# Graph Engine Constants
GRAPH_TRANSITION_WINDOW = 5     # Max submissions between A -> B to count as transition
MIN_TRANSITION_COUNT = 10       # Min users taking path to be valid edge

# Recommendation Quality Thresholds
MIN_ATT_SCORE = 5               # minimum rating gain to recommend
MIN_PROBABILITY_UPLIFT = 0.02   # minimum probability gain (2%)
P_VALUE_THRESHOLD = 0.01        # significance level (1%)

# Dataset Sources
# Kaggle: https://www.kaggle.com/datasets/intrincantation/cf-userdata
# HuggingFace: https://huggingface.co/datasets/denkCF/UsersCodeforcesSubmissionsEnd2024
# Schema Reference: https://codeforces.com/blog/entry/136853
