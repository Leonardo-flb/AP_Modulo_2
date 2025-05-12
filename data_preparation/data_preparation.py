# Data preparation for feeding the models will be carried out in this file.

import numpy as np
import matplotlib.pyplot as plt 
import torch
from collections import Counter

from torch.utils.data import DataLoader
from torch.utils.data import random_split
from .GRSDataset import GRSDataset

def grs_prepare_data_loaders(partition, path_train, path_osats, transforms=None, batch_size=8, seed=42):
    '''
    Prepares DataLoaders for training, validation, and testing of the GRS dataset.
    The `partition` parameter defines the percentage of data used for training.
    10% of the full dataset is used for validation, the rest for testing.
    Reproducibility is ensured via a fixed random seed.
    '''
    # Set global random seed
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    dataset = GRSDataset(path_train, path_osats, num_frames=600, transforms=transforms)

    train_size = int(partition * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_data, temp_data = random_split(dataset, [train_size, len(dataset) - train_size], generator=generator)
    val_data, test_data = random_split(temp_data, [val_size, test_size], generator=generator)

    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_dl  = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    train_dl_all = DataLoader(train_data, batch_size=len(train_data), shuffle=True)
    val_dl_all   = DataLoader(val_data, batch_size=len(val_data), shuffle=True)
    test_dl_all  = DataLoader(test_data, batch_size=len(test_data), shuffle=True)

    return train_dl, val_dl, test_dl, train_dl_all, val_dl_all, test_dl_all


# FOR VISUALIZATION and ANALYSIS

def visualize_dataset(train, test, val):
    print(f"Casos de Treino: {len(train.dataset)}")
    print(f"Casos de Validação: {len(val.dataset)}")
    print(f"Casos de Teste: {len(test.dataset)}")

    x, y = next(iter(train)) 
    print(f"(TREINO) Shape tensor batch -> input: {x.shape}, output: {y.shape}")

    x, y = next(iter(val)) 
    print(f"(VAL) Shape tensor batch -> input: {x.shape}, output: {y.shape}")

    x, y = next(iter(test))
    print(f"(TESTE) Shape tensor batch -> input: {x.shape}, output: {y.shape}")

    print(f'Valor maximo:{torch.max(x)} Valor mínimo:{torch.min(x)}')

GRS_CLASSES = {
    0: "Novice",
    1: "Intermediate",
    2: "Proficient",
    3: "Specialist"
}

def _plot_class_distribution(train_vals, val_vals, test_vals, labels):
    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(8, 5))
    plt.bar(x - width, train_vals, width, label='Train', color='steelblue')
    plt.bar(x,         val_vals, width, label='Validation', color='orange')
    plt.bar(x + width, test_vals, width, label='Test', color='green')

    plt.xlabel('GRS Class')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution per Dataset Split')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def visualize_class_distribution(train_dl, test_dl, val_dl, plot=False):
    print('helllo')
    def count_labels_fast(dataset):
        return Counter([dataset[idx][1] for idx in range(len(dataset))])

    train_dist = count_labels_fast(train_dl.dataset)
    val_dist   = count_labels_fast(val_dl.dataset)
    test_dist  = count_labels_fast(test_dl.dataset)

    print("Distribuição de classes (GRS):")
    for cls_id, cls_name in GRS_CLASSES.items():
        print(f"{cls_name:12} | Train: {train_dist.get(cls_id,0):3} | Val: {val_dist.get(cls_id,0):3} | Test: {test_dist.get(cls_id,0):3}")

    if plot:
        labels = [GRS_CLASSES[i] for i in range(4)]
        train_vals = [train_dist.get(i, 0) for i in range(4)]
        val_vals   = [val_dist.get(i, 0) for i in range(4)]
        test_vals  = [test_dist.get(i, 0) for i in range(4)]
        _plot_class_distribution(train_vals, val_vals, test_vals, labels)