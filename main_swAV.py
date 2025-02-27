import argparse
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from src.data import get_datasetCML
from src.logging_config import logger
from src.model.model import (
    LSTMModel,
    LeNet,
    LeNet_5,
    resnet20,
    resnet32,
    resnet44,
    resnet56,
    resnet110,
    resnet1202,
)
from src.utils import *

# import for clustering
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

MODEL_DICT = {
    "LSTMModel": LSTMModel,
    "LeNet": LeNet,
    "LeNet_5": LeNet_5,
    "resnet20": resnet20,
    "resnet32": resnet32,
    "resnet44": resnet44,
    "resnet56": resnet56,
    "resnet110": resnet110,
    "resnet1202": resnet1202,
}

def cluster_adaptive(embeddings, labels, method="kmeans", num_clusters=10):
    """
    Perform clustering on embeddings using different methods and visualize with t-SNE.
    
    Args:
        embeddings: ndarray of shape (n_samples, n_features)
        labels: ndarray of shape (n_samples,)
        method: str, clustering method ("kmeans", "gmm", "dbscan", "hdbscan", "cosine")
        num_clusters: int, number of clusters for applicable methods
    """
    
    if method == "kmeans":
        cluster_model = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = cluster_model.fit_predict(embeddings)

    elif method == "gmm":
        cluster_model = GaussianMixture(n_components=num_clusters, random_state=42)
        clusters = cluster_model.fit_predict(embeddings)

    elif method == "dbscan":
        cluster_model = DBSCAN(eps=0.5, min_samples=5)
        clusters = cluster_model.fit_predict(embeddings)

    elif method == "hdbscan":
        cluster_model = hdbscan.HDBSCAN(min_cluster_size=10)
        clusters = cluster_model.fit_predict(embeddings)

    elif method == "cosine":
        similarity_matrix = cosine_similarity(embeddings)
        cluster_model = AgglomerativeClustering(n_clusters=num_clusters, affinity="precomputed", linkage="average")
        clusters = cluster_model.fit_predict(1 - similarity_matrix)

    else:
        raise ValueError("Unsupported clustering method. Choose from 'kmeans', 'gmm', 'dbscan', 'hdbscan', 'cosine'.")

    # Giảm chiều dữ liệu bằng t-SNE để vẽ đồ thị
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_data = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(f"t-SNE Visualization of Feature Prototypes ({method})")
    plt.show()

    return clusters

def train_one_epoch(model, trainloader, optimizer, criterion, device, epoch):
    """
    Train one epoch for any given model on CIFAR-10 or CIFAR-100.
    """
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    embeddings = []
    labels_list = []

    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}", leave=True)

    for idx, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs, features = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        embeddings.append(features.cpu().detach().numpy())
        labels_list.append(labels.cpu().detach().numpy())

    epoch_loss = running_loss / len(trainloader)
    accuracy = 100 * correct / total

    logger.info(f"Epoch [{epoch+1}] | Loss: {epoch_loss:.4f} | Accuracy: {accuracy:.2f}%")

    return np.concatenate(embeddings, axis=0), np.concatenate(labels_list, axis=0)

def evaluate(model, testloader, criterion, device):
    """
    Evaluate the model on the test set and return loss, accuracy, precision, recall, and F1-score.
    """
    model.eval()
    total_loss = 0.0
    correct, total = 0, 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    avg_loss = total_loss / len(testloader)
    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    logger.info(f"Test Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}% | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-score: {f1:.4f}")
    
    return avg_loss, accuracy, precision, recall, f1

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Lấy mô hình từ argument
    model_cls = MODEL_DICT.get(args.model, None)
    if model_cls is None:
        raise ValueError(f"Model '{args.model}' is not supported!")

    model = model_cls(num_classes=args.num_classes).to(device)
    trainloader, testloader = get_datasetCML(args.dataset, batch_size=args.batch_size, num_class=100)

    # print_dataloader_info(trainloader)

    # Chọn optimizer
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:
        raise ValueError("Optimizer must be 'adam' or 'sgd'")

    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        train_one_epoch(model, trainloader, optimizer, criterion, device, epoch)
        if epoch == args.epochs - 1:  # Kiểm tra nếu là epoch cuối cùng
            torch.save(model.state_dict(), f"{args.model}_{args.dataset}_final.pth")
            print(f"Model saved at {args.model}_final.pth")
    
    evaluate(model, testloader, criterion, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SELF-SUPERVISED LEARNING")

    parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100"], default="cifar10", help="Dataset to use (cifar10 or cifar100)")
    parser.add_argument('--num_classes', type=int, default=10, help='Number of class in output')
    parser.add_argument("--model", type=str, choices=MODEL_DICT.keys(), default="resnet20", help="Model architecture")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--optimizer", type=str, choices=["adam", "sgd"], default="adam", help="Optimizer (adam or sgd)")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda", help="Device to use")

    args = parser.parse_args()
    main(args)


