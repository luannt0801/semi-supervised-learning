import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, ConcatDataset,TensorDataset
from tqdm import tqdm
from src.data import data_semi_learning
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

def train_one_epoch(model, trainloader, optimizer, criterion, device, epoch):
    """
    Train one epoch for any given model on CIFAR-10 or CIFAR-100.
    """
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}", leave=True)

    for idx, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)
        # inputs, labels = inputs.to(device), torch.tensor(labels).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(trainloader)
    accuracy = 100 * correct / total

    print(f"Epoch [{epoch+1}] | Loss: {epoch_loss:.4f} | Accuracy: {accuracy:.2f}%")


def train_with_pseudo_labels(args):

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_cls = MODEL_DICT.get(args.model, None)
    if model_cls is None:
        raise ValueError(f"Model '{args.model}' is not supported!")

    model = model_cls(num_classes=args.num_classes).to(device)

    trainset, unlabel_trainset = data_semi_learning(args.dataset, batch_size=args.batch_size, num_class=10, labeled_classes=[0,1,2,3,4])

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    optimizer = optim.Adam(model.parameters(), lr=args.lr) if args.optimizer == "adam" else optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    print("==> Initial training with labeled dataset...")
    for epoch in range(args.epochs):
        train_one_epoch(model, trainloader, optimizer, criterion, device, epoch)

    print("==> Generating pseudo labels from unlabeled data...")
    unlabel_loader = DataLoader(unlabel_trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    pseudo_dataset = generate_pseudo_labels(model, unlabel_loader, device)

    if pseudo_dataset is not None:
        print(f"Number of samples with pseudo labels: {len(pseudo_dataset)}")

        combined_dataset = ConcatDataset([trainset, pseudo_dataset])
        final_train_loader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

        print(f"Type of pseudo_dataset: {type(combined_dataset)}")
        if isinstance(combined_dataset, torch.utils.data.Dataset):
            print(f"Sample from pseudo_dataset: {combined_dataset[0]}")


        print("==> Retrain with extended data (including pseudo labels)...")
        for epoch in range(args.epochs):
            train_one_epoch(model, final_train_loader, optimizer, criterion, device, epoch)

    torch.save(model.state_dict(), f"model_save/{args.model}_{args.dataset}_pseudo_final.pth")
    print(f"Model saved at {args.model}_{args.dataset}_pseudo_final.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SELF-SUPERVISED LEARNING")

    parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100"], default="cifar10", help="Dataset to use")
    parser.add_argument('--num_classes', type=int, default=10, help='Number of class in output')
    parser.add_argument("--model", type=str, choices=MODEL_DICT.keys(), default="resnet20", help="Model architecture")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--optimizer", type=str, choices=["adam", "sgd"], default="adam", help="Optimizer")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda", help="Device to use")

    args = parser.parse_args()
    train_with_pseudo_labels(args)




