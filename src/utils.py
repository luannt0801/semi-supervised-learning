import torch
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from collections import Counter

def show_image(sample: torch.Tensor, label: int):
    if isinstance(sample, torch.Tensor):
        print(f"\nSize of one sample: {sample.shape}")
    print(f"\nLabel of the first sample: {label}")

    if isinstance(sample, torch.Tensor) and sample.ndim in [2,3]:
        plt.imshow(sample.permute(1, 2, 0) if sample.ndim == 3 else sample, cmap ="gray")
        plt.title(f"Label: {label}")
        plt.axis("off")
        plt.show()

# def print_dataset_info(dataset):
#     """
#     Input the dataset
#     """
#     print(f"Number sample of dataset: {len(dataset)}")
#     labels = [dataset[i] for i in range(len(dataset))]
#     label_counts = Counter(labels)
#     print(f"\nNumber label of dataset: {label_counts}")
#     for label, count in sorted(label_counts.items):
#         print(f"\n- Label {label}: {count} samples")

def print_dataset_info(dataset):
    """
    Dataset Info
    """
    print(f"\nTotal samples in dataset: {len(dataset)}")

    labels = [dataset[i][1] for i in range(len(dataset))]  
    
    label_counts = Counter(labels)

    print("\nLabel distribution in dataset:")
    for label, count in sorted(label_counts.items()):
        print(f"\n- Label {label}: {count} samples")

    

def print_dataloader_info(dataloader):
    """
    Print information about the given DataLoader, including:
    - Total number of batches
    - Total number of samples
    - Unique labels and their counts
    - Sample batch information
    """
    num_batches = len(dataloader)
    print(f"🔹 Total number of batches: {num_batches}")

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        label_counts = Counter(labels.tolist())

        print(f"\n📌 Batch {batch_idx + 1}:")
        print(f"  - Number of samples in batch: {len(labels)}")
        print(f"  - Number of unique labels in batch: {len(label_counts.keys())}")
        print(f"  - Distribution of samples per label: {dict(label_counts)}")

def create_dir(_dir):
    """
    Creates given directory if it is not present.
    """
    if not os.path.exists(_dir):
        os.makedirs(_dir)

