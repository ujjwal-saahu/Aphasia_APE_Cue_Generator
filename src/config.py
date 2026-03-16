from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ============================================================
# DATA FOLDERS
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"

DATA_RAW = DATA_DIR / "raw"
DATA_INTERIM = DATA_DIR / "interim"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_FINAL = DATA_DIR / "final"

# ============================================================
# RAW DATA
# ============================================================

RAW_DATA_PATH = DATA_RAW / "Raw_Data.csv"

# ============================================================
# PROCESSED DATA
# ============================================================

CLEAN_DATA_PATH = DATA_PROCESSED / "clean_dataset.csv"

BEST_HUMAN_CUES_PATH = DATA_PROCESSED / "best_human_cues.csv"

SEED_EXAMPLES_PATH = DATA_PROCESSED / "seed_examples.csv"

# ============================================================
# CATEGORY PROMPT PIPELINE
# ============================================================

CATEGORY_PROMPT_CANDIDATES_PATH = DATA_INTERIM / "category_prompt_candidates.csv"

CATEGORY_INITIAL_RESULTS_PATH = DATA_INTERIM / "category_initial_generation_results.csv"

CATEGORY_ROUND0_SUMMARY_PATH = DATA_INTERIM / "category_round0_prompt_summary.csv"

CATEGORY_APE_ALL_RESULTS_PATH = DATA_INTERIM / "category_ape_all_round_results.csv"

CATEGORY_APE_ALL_SUMMARY_PATH = DATA_INTERIM / "category_ape_all_round_prompt_summary.csv"

BEST_PROMPT_BY_CATEGORY_PATH = DATA_FINAL / "best_prompt_by_category.csv"

FINAL_GENERATED_CUES_BY_CATEGORY_PATH = DATA_FINAL / "final_generated_cues_by_category.csv"