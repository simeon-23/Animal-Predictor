"""Train the animal classifier."""

import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from model_utils import build_model, get_transform, log_metric, save_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--from_scratch", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(42)
    random.seed(42)

    dataset = datasets.ImageFolder(args.train_data, transform=get_transform())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = build_model(len(dataset.classes), pretrained=not args.from_scratch, freeze=not args.from_scratch)
    if not args.from_scratch:
        model.features.eval()
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    loss_function = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        if not args.from_scratch:
            model.features.eval()
        total_loss = 0.0
        correct = 0
        for images, labels in loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
        average_loss = total_loss / len(dataset)
        accuracy = correct / len(dataset)
        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {average_loss:.4f} - train accuracy: {accuracy:.4f}")
        log_metric("train_loss", average_loss, epoch + 1)
        log_metric("train_accuracy", accuracy, epoch + 1)
    save_model(model, dataset.classes, args.model_dir)
    print(f"Saved model to {Path(args.model_dir).resolve()}")


if __name__ == "__main__":
    main()