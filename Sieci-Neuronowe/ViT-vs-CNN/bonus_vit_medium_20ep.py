import torch
import torch.nn as nn
from Vision_Transformer import VisionTransformer
from data_loader_imagenet import data_loader_imagenet
import time
import json
from cnn_model import count_parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100 * correct / total


def run_vit_medium_20ep():
    config = {
        'patch_size': 4, 'embed_dim': 256, 'num_heads': 8,
        'Nx': 6, 'mlp_dim': 1024, 'epochs': 20, 'lr': 3e-4
    }

    train_loader, test_loader = data_loader_imagenet('./data', batch_size=128)

    model = VisionTransformer(
        img_size=32,
        patch_size=config['patch_size'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        Nx=config['Nx'],
        mlp_dim=config['mlp_dim'],
        n_class=10,
        dropout=0.1
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    num_params = count_parameters(model)
    history = []
    start_time = time.time()

    print(f"\n=== ViT_Medium 20 epok — Imagenette ===")
    print(f"Parametry: {num_params:,} | embed_dim={config['embed_dim']} | Nx={config['Nx']} | heads={config['num_heads']}")

    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        train_acc = 100 * correct_train / total_train
        test_acc = evaluate(model, test_loader)
        history.append({
            "epoch": epoch + 1,
            "loss": running_loss / len(train_loader),
            "train_acc": train_acc,
            "test_acc": test_acc
        })
        print(f"  Epoka {epoch+1:2d}/20 | Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")

    train_time = time.time() - start_time
    best_epoch = max(history, key=lambda x: x['test_acc'])

    result = {
        "model": "ViT",
        "experiment": "ViT_Medium_20ep",
        "dataset": "Imagenette2-320",
        "config": config,
        "num_params": num_params,
        "test_acc": history[-1]['test_acc'],
        "train_acc": history[-1]['train_acc'],
        "best_test_acc": best_epoch['test_acc'],
        "best_epoch": best_epoch['epoch'],
        "train_time_s": train_time,
        "time_per_epoch": train_time / config['epochs'],
        "history": history,
        "vs_10ep_imagenet": 63.16,   # z poprzedniego runu
        "vs_10ep_cifar10": 68.75     # z poprzedniego runu
    }

    with open("bonus.json", "w", encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"\n{'='*55}")
    print(f"WYNIK KOŃCOWY: {result['test_acc']:.2f}% (epoka 20)")
    print(f"NAJLEPSZY:     {result['best_test_acc']:.2f}% (epoka {result['best_epoch']})")
    print(f"vs ViT_Medium 10ep ImageNet: {result['vs_10ep_imagenet']}%  → "
          f"{'+'if result['best_test_acc'] > result['vs_10ep_imagenet'] else ''}"
          f"{result['best_test_acc'] - result['vs_10ep_imagenet']:.2f}%")
    print(f"Czas: {train_time:.1f}s  ({result['time_per_epoch']:.1f}s/epoka)")
    print(f"Wyniki zapisane do: bonus.json")


if __name__ == "__main__":
    run_vit_medium_20ep()
