import os
from pathlib import Path

SRC_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = SRC_DIR.parent
DATA_DIR = ROOT_DIR / 'dataset'
EXP_DIR = ROOT_DIR / 'experiments'
KAGGLE_EVAL_DIR = ROOT_DIR / 'kaggle_evaluation'
GLOBAL_SEED = 42