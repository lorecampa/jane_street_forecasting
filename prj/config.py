import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

SRC_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = SRC_DIR.parent
EXP_DIR = ROOT_DIR / 'experiments'
KAGGLE_EVAL_DIR = ROOT_DIR / 'kaggle_evaluation'
GLOBAL_SEED = 42

data_dir_env = os.getenv('DATA_DIR')
if data_dir_env:
    DATA_DIR = Path(data_dir_env)
else:
    DATA_DIR = ROOT_DIR / 'dataset'

