import os
import logging
from lightgbm import Dataset
from datetime import datetime
import argparse
import pandas as pd
import numpy as np
from prj.config import DATA_DIR, EXP_DIR
from prj.data import DATA_ARGS_CONFIG
from prj.data.data_loader import DataLoader
from prj.logger import get_default_logger

def main(output_dir, num_bins, logger: logging.Logger): 
    data_args = DATA_ARGS_CONFIG['lgbm']
    loader = DataLoader(data_dir=DATA_DIR, **data_args)
    
    X, y, w, _ = loader.load_partitions(0, 8)
    features = loader.features
    categorical_features = []
    
    logger.info(f'Features ({len(features)}): {np.array(features)}')
    logger.info(f'Categorical features: {np.array(categorical_features)}')
    logger.info(f'Creating lightgbm dataset with {num_bins} bins')
    
    
    lgbm_dataset = Dataset(
        pd.DataFrame(X, columns=features).astype({c: 'category' for c in categorical_features}),
        label=y,
        weight=w,
        feature_name=features,
        categorical_feature=categorical_features,
        params={'max_bin': num_bins},
    )
    lgbm_dataset.construct()

    train_dir = os.path.join(output_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)
    _path = os.path.join(train_dir, "lgbm_dataset.bin")
    logger.info(f'Saving converted train dataset at: {_path}')
    lgbm_dataset.save_binary(_path)
    
    
if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = str(
        EXP_DIR / f"lgbm_dataset_{timestamp}"
    )
    NUM_BINS = 128
    logger = get_default_logger()
        
    main(OUTPUT_DIR, NUM_BINS, logger)