import torch
import torch.nn as nn
from Vision_Transformer import VisionTransformer
from data_loader_imagenet import data_loader_imagenet
import time
from cnn_model import CifarCNN, count_parameters
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def run_training_loop(model, config):
    train_loader, test_loader = data_loader_imagenet(
        './data',
        batch_size=config.get('batch_size', 128)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    num_params = count_parameters(model)
    history = []
    start_time = time.time()

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
        epoch_data = {
            "epoch": epoch + 1,
            "loss": running_loss / len(train_loader),
            "train_acc": train_acc,
            "test_acc": test_acc
        }
        history.append(epoch_data)
        print(f"  Epoka {epoch+1}/{config['epochs']} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    train_time = time.time() - start_time
    return {
        "test_acc": test_acc,
        "train_acc": train_acc,
        "train_time_s": train_time,
        "num_params": num_params,
        "history": history
    }


def run_ViT(config):
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
    return run_training_loop(model, config)


def run_cnn(config):
    model = CifarCNN().to(device)
    return run_training_loop(model, config)


if __name__ == "__main__":
    results = []

    # 1. CNN epoki (5 i 10) - identyczne jak CIFAR-10
    for e in [5, 10]:
        print(f"\n=== CNN epochs={e} (Imagenette) ===")
        res_cnn = run_cnn({'epochs': e})
        results.append({
            'model': 'CNN',
            'experiment': 'epoki',
            'param_value': e,
            **res_cnn,
            'time_per_epoch': res_cnn['train_time_s'] / e
        })
        print(f"CNN {e} epok: {res_cnn['test_acc']:.2f}% | {res_cnn['num_params']:,} params | {res_cnn['train_time_s']:.1f}s")

    # 2. ViT Small (5 epok) - Nx=3, num_heads=4, embed_dim=128, mlp_dim=256
    cfg_small = {
        'patch_size': 4, 'embed_dim': 128, 'num_heads': 4,
        'Nx': 3, 'mlp_dim': 256, 'epochs': 5, 'lr': 1e-3
    }
    print(f"\n=== ViT Small (5 epok, Imagenette) ===")
    res_small = run_ViT(cfg_small)
    results.append({
        'model': 'ViT',
        'experiment': 'ViT_Small',
        'param_value': 5,
        **res_small,
        'time_per_epoch': res_small['train_time_s'] / 5
    })
    print(f"ViT Small: {res_small['test_acc']:.2f}% | {res_small['num_params']:,} params | {res_small['train_time_s']:.1f}s")

    # 3. ViT Medium (10 epok) - Nx=6, num_heads=8, embed_dim=256, mlp_dim=1024
    cfg_medium = {
        'patch_size': 4, 'embed_dim': 256, 'num_heads': 8,
        'Nx': 6, 'mlp_dim': 1024, 'epochs': 10, 'lr': 1e-3
    }
    print(f"\n=== ViT Medium (10 epok, Imagenette) ===")
    res_medium = run_ViT(cfg_medium)
    results.append({
        'model': 'ViT',
        'experiment': 'ViT_Medium',
        'param_value': 10,
        **res_medium,
        'time_per_epoch': res_medium['train_time_s'] / 10
    })
    print(f"ViT Medium: {res_medium['test_acc']:.2f}% | {res_medium['num_params']:,} params | {res_medium['train_time_s']:.1f}s")

    # 4. ViT Large (10 epok) - Nx=9, num_heads=8, embed_dim=256, mlp_dim=1024
    cfg_large = {
        'patch_size': 4, 'embed_dim': 256, 'num_heads': 8,
        'Nx': 9, 'mlp_dim': 1024, 'epochs': 10, 'lr': 1e-3
    }
    print(f"\n=== ViT Large (10 epok, Imagenette) ===")
    res_large = run_ViT(cfg_large)
    results.append({
        'model': 'ViT',
        'experiment': 'ViT_Large',
        'param_value': 10,
        **res_large,
        'time_per_epoch': res_large['train_time_s'] / 10
    })
    print(f"ViT Large: {res_large['test_acc']:.2f}% | {res_large['num_params']:,} params | {res_large['train_time_s']:.1f}s")

    # 5. ViT Best (10 epok) - patch_size=2, Nx=6, num_heads=8, embed_dim=256, mlp_dim=1024
    cfg_best = {
        'patch_size': 2, 'embed_dim': 256, 'num_heads': 8,
        'Nx': 6, 'mlp_dim': 1024, 'epochs': 10, 'lr': 1e-3
    }
    print(f"\n=== ViT Best patch_size=2 (10 epok, Imagenette) ===")
    res_best = run_ViT(cfg_best)
    results.append({
        'model': 'ViT',
        'experiment': 'ViT_Best',
        'param_value': 10,
        **res_best,
        'time_per_epoch': res_best['train_time_s'] / 10
    })
    print(f"ViT Best: {res_best['test_acc']:.2f}% | {res_best['num_params']:,} params | {res_best['train_time_s']:.1f}s")

    output_file = "wyniki_eksperymentow_imagenet.json"
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\nWyniki zapisane do {output_file}")
