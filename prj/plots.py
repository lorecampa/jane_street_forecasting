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