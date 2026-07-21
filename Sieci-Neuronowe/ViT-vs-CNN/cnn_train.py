import torch
import torch.nn as nn
import torch.optim as optim
import time
from data_loader import data_loader
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total


def train(model, epochs=10, batch_size=128, lr=0.001):
    train_loader, test_loader = data_loader(path="./data", batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Rozpoczynam trening")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        test_acc = evaluate(model, test_loader)
        print(
            f"Epoka [{epoch+1}/{epochs}] "
            f"| Loss: {running_loss/len(train_loader):.4f}"
            f"| Train Accuracy: {100.*correct/total:.2f}%"
            f"| Test Accuracy: {test_acc:.2f}%"
        )

    end_time = time.time()
    print(f"\nTrening zakończony w czasie: {end_time - start_time:.2f}s")
