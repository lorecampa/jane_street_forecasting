import os
import logging
from lightgbm import Dataset
from datetime import datetime
import argparse
import pandas as pd
import numpy as np
from prj.config import DATA_DIR, EXP_DIR
from prj.data import DATA_ARGS_CONFIG
from prj.data.data_loader import PARTITIONS_DATE_INFO, DataConfig, DataLoader
from prj.logger import get_default_logger

def get_cli_args():
    """Create CLI parser and return parsed arguments"""
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--data_dir',
        type=int,
        default=DATA_DIR,
    )
    
    parser.add_argument(
        '--start_dt',
        type=int,
        default=PARTITIONS_DATE_INFO[5],
    )
    parser.add_argument(
        '--end_dt',
        type=int,
        default=PARTITIONS_DATE_INFO[9],
    )
    
    parser.add_argument(
        '--max_bin',
        type=int,
        default=128,
    )
    return parser.parse_args()


def main(output_dir, data_dir, num_bins, start_dt, end_dt, logger: logging.Logger): 
    
    config = DataConfig(
        include_lags=False,
        zero_fill=False,
        ffill=False,            
    )
    loader = DataLoader(data_dir=data_dir, config=config)
    
    X, y, w, _ = loader.load_numpy(start_dt=start_dt, end_dt=end_dt)
    
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
        params={'max_bin': num_bins, 'feature_pre_filter': False},
    )
    lgbm_dataset.construct()

    train_dir = os.path.join(output_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)
    _path = os.path.join(train_dir, "lgbm_dataset.bin")
    logger.info(f'Saving converted train dataset at: {_path}')
    lgbm_dataset.save_binary(_path)
    
    
if __name__ == '__main__':
    args = get_cli_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = str(
        EXP_DIR / f"lgbm_dataset_{args.start_dt}_{args.end_dt}{timestamp}"
    )
    logger = get_default_logger()
        
    main(OUTPUT_DIR, data_dir=args.data_dir, num_bins=args.max_bin, start_dt=args.start_dt, end_dt=args.end_dt, logger=logger)