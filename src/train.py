"""Train the classifier."""
import argparse
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
    dataset = datasets.ImageFolder(args.train_data, transform=get_transform())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = build_model(len(dataset.classes), pretrained=not args.from_scratch, freeze=not args.from_scratch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        model.train()
        if not args.from_scratch:
            model.features.eval()
        correct = total = running = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * labels.size(0)
            correct += (output.argmax(1) == labels).sum().item()
            total += labels.size(0)
        loss_value, accuracy = running / total, correct / total
        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {loss_value:.4f} - train accuracy: {accuracy:.4f}")
        log_metric("loss", loss_value, epoch + 1)
        log_metric("train_accuracy", accuracy, epoch + 1)
    save_model(model.cpu(), dataset.classes, args.model_dir)
    print(f"Saved model to {Path(args.model_dir).resolve()}")


if __name__ == "__main__":
    main()