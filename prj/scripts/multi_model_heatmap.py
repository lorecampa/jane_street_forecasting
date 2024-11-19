
import numpy as np
from prj.config import EXP_DIR
import os
from prj.plots import plot_partition_heatmap
from prj.utils import load_pickle

def main():
    OUT_DIR = EXP_DIR / 'multi_model_heatmap'
    os.makedirs(OUT_DIR, exist_ok=True)
    FIRST_EVAL_PARTITION = 0
    METRICS = ['r2_w', 'rmse_w', 'mae_w', 'mse_w']
    PATH = EXP_DIR / 'train'
    
    
    experiments = sorted([exp for exp in os.listdir(PATH) if exp.startswith('lgbm')],
                         key=lambda x: int(x.split('_')[1].split('-')[0]))
    y_tick_labels = ['_'.join(exp.split('_')[:-2]) for exp in experiments]
    x_tick_labels = [f'partition {i}' for i in range(10)]
    
    for metric in METRICS:
        res = []
        for exp in experiments:
            exp_eval_dir = EXP_DIR / 'train' / exp / 'evaluations'
            exp_eval_dict = load_pickle(exp_eval_dir / 'result_aggregate.json')
            res.append(np.array(exp_eval_dict[metric]))
        
        training_partitions = []
        for exp in experiments:
            exp_first_partition = int(exp.split('_')[1].split('-')[0])
            exp_last_partition = int(exp.split('_')[1].split('-')[1])
            training_partitions.append([(exp_first_partition-FIRST_EVAL_PARTITION, exp_last_partition-FIRST_EVAL_PARTITION)])
            
        plot_partition_heatmap(
            np.array(res),
            training_partitions, 
            title=f'{metric.upper()} Heatmap', 
            xticklabels=x_tick_labels, 
            yticklabels=y_tick_labels, 
            save_path=os.path.join(OUT_DIR, f'{metric}_heatmap.png'),
            decimal_places=4,
            invert_scale=metric not in ['r2_w'] # Invert scale for loss metrics
        )
    
    
    


if __name__ == '__main__':
    main()