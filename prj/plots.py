import gc
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_histogram(data: np.ndarray) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Distribution of Values (Histogram)', fontsize=16)
    plt.xlabel('Value', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, alpha=0.5)
    plt.show()

def plot_density(data: np.ndarray) -> None:
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data, fill=True, color='skyblue', alpha=0.7)
    plt.title('Distribution of Values (Density Plot)', fontsize=16)
    plt.xlabel('Value', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.grid(True, alpha=0.5)
    plt.show()
    


def plot_time_series(data: np.ndarray, labels: List[str], times: np.ndarray, title:str = "") -> None:
    plt.figure(figsize=(10, 6))
    for i in range(data.shape[1]):
        plt.plot(times, data[:, i], label=labels[i])
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
    

def plot_partition_heatmap(results: np.ndarray, training_partitions: list[list[tuple[int, int]]], title="Partition Heatmap", 
                 xticklabels: list[str]=None, yticklabels: list[str]=None,
                 decimal_places: int = 2, use_e_notation: bool = False, invert_scale:bool = False,
                 show_best_per_partition: bool = False,
                 save_path:str = None, force_show: bool = False):
    assert results.ndim == 2
    assert results.shape[0] == len(training_partitions)
    
    results = np.array(results)
    num_rows, num_cols = results.shape
    plt.figure(figsize=(num_cols*1.5, num_rows*1.5))
    cmap = sns.light_palette("blue", as_cmap=True)
    if invert_scale:
        cmap = cmap.reversed()
    
    fmt = f".{decimal_places}{'e' if use_e_notation else 'f'}"
    
    ax = sns.heatmap(results, annot=True, fmt=fmt, cmap=cmap, cbar=True, linewidths=0.5, linecolor='black', 
                     xticklabels=[f'Day {i+1}' for i in range(num_cols)] if xticklabels is None else xticklabels, 
                     yticklabels=[f'Results {i+1}' for i in range(num_rows)] if yticklabels is None else yticklabels)
    
    for row, train_days in enumerate(training_partitions):
        for start, end in train_days:
            ax.add_patch(plt.Rectangle((start, row), end - start + 1, 1, fill=False, edgecolor='red', lw=3))
    
    if show_best_per_partition:
        best_indices = []
        for col in range(results.shape[1]):
            best_rows = np.argsort(results[:, col]) if invert_scale else np.argsort(results[:, col])[::-1]            
            best_rows = [x for x in best_rows if x != col-1] # Exclude the same partition, should be changed
            if len(best_rows) > 0:
                best_indices.append((best_rows[0], col))
            
        for best_row, best_col in best_indices:
            ax.add_patch(plt.Rectangle((best_col, best_row), 1, 1, fill=False, edgecolor='orange', lw=3))
        
    plt.title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        if force_show:
            plt.show()
    else:
        plt.show()
    plt.close()
    gc.collect()