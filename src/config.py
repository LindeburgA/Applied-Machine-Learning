"""All project decisions in one place. IDs (D1, D4a ...) refer to the Decision Log."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("PLAYING_TIME_DATA_DIR", ROOT / "data"))
OUT_DIR = ROOT / "output"     # everything the scripts produce

# Leagues (D10). DK1 is always the test league.
TARGET_LEAGUE = "DK1"
EXTRA_LEAGUES = ["NL1", "BE1", "PO1", "SC1"]
ALL_LEAGUES = [TARGET_LEAGUE] + EXTRA_LEAGUES

# Timeline (D1-D3): snapshot on 1 July of season s; features use data dated strictly before it.
SNAPSHOT_MONTH, SNAPSHOT_DAY = 7, 1
PRE_SEASON_END = (8, 31)          # moves until 31 Aug count as "left_pre_season" (D7)
FIRST_SNAPSHOT, LAST_SNAPSHOT = 2014, 2025   # lineups start mid-2013, so 2013 has no prior-season squads

# Population (D4c)
AGE_MIN, AGE_MAX = 17.0, 22.0     # 17.0 <= age < 22.0  (= 17 to 21 years)

# Target (D5-D6)
THRESHOLD = 0.30
SENSITIVITY_THRESHOLDS = [0.20, 0.40]

# Values (D9.3)
VALUE_MAX_AGE_DAYS = 365

# Academy promotions (D4e): youth or reserve team names in transfers
YOUTH_PATTERN = r"\bU ?(?:1[5-9]|2[0-3])\b|Youth|Yth\.?|Academy|^Jong |\bII$| B$"

# Validation (D11)
SELECTION_TEST_SEASONS = [2020, 2021, 2022, 2023]
HOLDOUT_TEST_SEASONS = [2024, 2025]

# Decision threshold (D13): keep if p >= c / (c + 1)
COST_RATIOS = [1.5, 2.0, 3.0]
MAIN_COST_RATIO = 2.0

RANDOM_STATE = 42
N_BOOTSTRAP = 1000
