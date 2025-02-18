import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import os
import random
import string
import torch
from collections import Counter
from torchvision import datasets, transforms
from torch.utils.data import Subset
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

__all__ = [
    "get_datasetCML"
]

def pad_sequences(encoded_domains, maxlen):
    domains = []
    for domain in encoded_domains:
        if len(domain) >= maxlen:
            domains.append(domain[:maxlen])
        else:
            domains.append([0] * (maxlen - len(domain)) + domain)
    return np.asarray(domains)

def get_datasetCML(data_use, num_class, batch_size):
    if data_use == 'cifar10':
        path = "D:\\2025\\Projects\\Federated Learning with the raw meat in the dish\\data\\images"
        transform_train = transforms.Compose([transforms.RandomGrayscale(0.2),
                                        transforms.RandomHorizontalFlip(0.5),
                                        transforms.RandomVerticalFlip(0.2),
                                        transforms.RandomRotation(30),
                                        transforms.RandomAdjustSharpness(0.4),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                        ])
        transform_test = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
                                    )
        
        trainset = datasets.CIFAR10(root=path, train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root=path, train=False,
                                    download=True, transform=transform_test)
    
        trainloader = DataLoader(trainset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
        testloader = DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
        
    elif data_use == "cifar100":
        input_size = 32
        path = "D:\\2025\\Projects\\Federated Learning with the raw meat in the dish\\data\\images"
        transform_train = transforms.Compose(
                        [
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomCrop(32, 4),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                        ])
        transform_test = transforms.Compose([
                                transforms.Resize(input_size),
                                transforms.CenterCrop(input_size),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),]
                        )
        trainset = datasets.CIFAR100(root=path, train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(root=path, train=False, download=True, transform=transform_test)

        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    elif data_use == 'dga':
        data_folder = path = "D:\\2025\\Projects\\Federated Learning with the raw meat in the dish\\data\\dga"
        dga_types = [dga_type for dga_type in os.listdir(data_folder) if os.path.isdir(f"{data_folder}/{dga_type}")]
        # print(f"Detected DGA types: {dga_types}")
        my_df = pd.DataFrame(columns=['domain', 'type', 'label'])

        for dga_type in dga_types:
            files = os.listdir(f"{data_folder}/{dga_type}")
            for file in files:
                # num_labels += 1
                with open(f"{data_folder}/{dga_type}/{file}", 'r') as fp:
                    lines = fp.readlines()

                    if num_class == 2:
                        domains_with_type = [[line.strip(), dga_type, 1] for line in lines]
                    elif num_class == 11:
                        label_index = dga_types.index(dga_type) + 1
                        domains_with_type = [[line.strip(), dga_type, label_index] for line in lines]
                    elif num_class == 10:
                        label_index = dga_types.index(dga_type) + 1
                        domains_with_type = [[line.strip(), dga_type, label_index] for line in lines]
                    else:
                        raise ValueError("Please input the correct number of labels for DGA data!")

                    appending_df = pd.DataFrame(domains_with_type, columns=['domain', 'type', 'label'])
                    my_df = pd.concat([my_df, appending_df], ignore_index=True)

        with open(f'{data_folder}/benign.txt', 'r') as fp:
            benign_lines = fp.readlines()[:]
            domains_with_type = [[line.strip(), 'benign', 0] for line in benign_lines]
            appending_df = pd.DataFrame(domains_with_type, columns=['domain', 'type', 'label'])
            my_df = pd.concat([my_df, appending_df], ignore_index=True)
        
        # Pre-processing
        domains = my_df['domain'].to_numpy()
        labels = my_df['label'].to_numpy()

        char2ix = {x: idx + 1 for idx, x in enumerate(string.printable)}
        ix2char = {ix: char for char, ix in char2ix.items()}

        encoded_domains = [[char2ix[y] for y in x if y in char2ix] for x in domains]
        encoded_labels = labels  # Giữ nguyên nhãn từ dữ liệu

        encoded_labels = np.asarray([label for idx, label in enumerate(encoded_labels) if len(encoded_domains[idx]) > 1])
        encoded_domains = [domain for domain in encoded_domains if len(domain) > 1]

        assert len(encoded_domains) == len(encoded_labels)

        maxlen = max(len(domain) for domain in encoded_domains)  # Đặt chiều dài tối đa
        padded_domains = pad_sequences(encoded_domains, maxlen)

        X_train, X_test, y_train, y_test = train_test_split(padded_domains, encoded_labels, test_size=0.10, shuffle=True)
        trainset = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.Tensor(y_train))
        testset = TensorDataset(torch.tensor(X_test, dtype=torch.long), torch.Tensor(y_test))

        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader

def data_semi_learning(data_use, num_class, batch_size, labeled_classes=[0, 1, 2, 3, 4]):
    if data_use == 'cifar10':
        path = "D:\\2025\\Projects\\Federated Learning with the raw meat in the dish\\data\\images"
        transform_train = transforms.Compose([
            transforms.RandomGrayscale(0.2),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.2),
            transforms.RandomRotation(30),
            transforms.RandomAdjustSharpness(0.4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # Load CIFAR-10
        trainset = datasets.CIFAR10(root=path, train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root=path, train=False, download=True, transform=transform_test)

    elif data_use == "cifar100":
        input_size = 32
        path = "D:\\2025\\Projects\\Federated Learning with the raw meat in the dish\\data\\images"
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        # Load CIFAR-100
        trainset = datasets.CIFAR100(root=path, train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(root=path, train=False, download=True, transform=transform_test)

    # Chuyển targets sang NumPy để xử lý nhanh hơn
    targets = np.array(trainset.targets)

    # Lọc chỉ mục nhanh bằng NumPy
    labeled_indices = np.where(np.isin(targets, labeled_classes))[0]
    unlabeled_indices = np.where(~np.isin(targets, labeled_classes))[0]

    # Shuffle để tránh dữ liệu có nhãn tập trung
    np.random.shuffle(labeled_indices)
    np.random.shuffle(unlabeled_indices)

    # Chia tập dữ liệu
    labeled_set = Subset(trainset, labeled_indices)
    unlabeled_set = Subset(trainset, unlabeled_indices)

    return labeled_set, unlabeled_set