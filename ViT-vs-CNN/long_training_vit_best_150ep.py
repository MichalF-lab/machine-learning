import torch
import torch.nn as nn
from Vision_Transformer import VisionTransformer
from data_loader import data_loader
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


def run_vit_best_150ep(dataset_name, train_loader, test_loader):
    config = {
        'patch_size': 2, 'embed_dim': 256, 'num_heads': 8,
        'Nx': 6, 'mlp_dim': 1024, 'epochs': 150, 'lr': 3e-4
    }

    model = VisionTransformer(
        img_size=32,
        patch_size=config['patch_size'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        Nx=config['Nx'],
        mlp_dim=config['mlp_dim'],
        n_class=10,
        dropout=0.2
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)
    num_params = count_parameters(model)
    history = []
    start_time = time.time()

    print(f"\n{'='*65}")
    print(f"=== ViT_Best (patch_size=2) 150 epok — {dataset_name} ===")
    print(f"{'='*65}")
    print(f"Parametry: {num_params:,} | patch_size=2 | embed_dim={config['embed_dim']} | Nx={config['Nx']} | heads={config['num_heads']}")
    print(f"Tokeny: (32/2)² = 256 tokenów na obraz")

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

        scheduler.step()
        train_acc = 100 * correct_train / total_train
        test_acc = evaluate(model, test_loader)

        # Zapisuj co 10 epok (oszczędź RAM)
        if (epoch + 1) % 10 == 0:
            history.append({
                "epoch": epoch + 1,
                "loss": running_loss / len(train_loader),
                "train_acc": train_acc,
                "test_acc": test_acc
            })
            print(f"  Epoka {epoch+1:3d}/150 | Train: {train_acc:.2f}% | Test: {test_acc:.2f}%", flush=True)
        else:
            # Zapisuj ostatnią epokę
            if epoch == config['epochs'] - 1:
                history.append({
                    "epoch": epoch + 1,
                    "loss": running_loss / len(train_loader),
                    "train_acc": train_acc,
                    "test_acc": test_acc
                })

    train_time = time.time() - start_time
    best_epoch = max(history, key=lambda x: x['test_acc'])

    return {
        "dataset": dataset_name,
        "config": config,
        "num_params": num_params,
        "test_acc": history[-1]['test_acc'],
        "train_acc": history[-1]['train_acc'],
        "best_test_acc": best_epoch['test_acc'],
        "best_epoch": best_epoch['epoch'],
        "train_time_s": train_time,
        "time_per_epoch": train_time / config['epochs'],
        "history": history
    }


if __name__ == "__main__":
    # CIFAR-10
    print("\n" + "="*65)
    print("ETAP 1: CIFAR-10")
    print("="*65)
    c_train, c_test = data_loader('./data', batch_size=128)
    result_cifar = run_vit_best_150ep("CIFAR-10", c_train, c_test)

    # Imagenette
    print("\n" + "="*65)
    print("ETAP 2: ImageNet (Imagenette2-320)")
    print("="*65)
    i_train, i_test = data_loader_imagenet('./data', batch_size=128)
    result_inet = run_vit_best_150ep("Imagenette2-320", i_train, i_test)

    # Zapisz wyniki
    results = {
        "cifar10": result_cifar,
        "imagenet": result_inet,
        "summary": {
            "model": "ViT_Best (patch_size=2)",
            "epochs": 150,
            "best_cifar_acc": result_cifar['best_test_acc'],
            "best_cifar_epoch": result_cifar['best_epoch'],
            "best_inet_acc": result_inet['best_test_acc'],
            "best_inet_epoch": result_inet['best_epoch'],
            "total_time_s": result_cifar['train_time_s'] + result_inet['train_time_s']
        }
    }

    with open("long_training_vit_best_150ep.json", "w", encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n{'='*65}")
    print("PODSUMOWANIE")
    print(f"{'='*65}")
    print(f"CIFAR-10:       {result_cifar['best_test_acc']:.2f}% (epoka {result_cifar['best_epoch']}) | {result_cifar['train_time_s']/3600:.1f}h")
    print(f"Imagenette:     {result_inet['best_test_acc']:.2f}% (epoka {result_inet['best_epoch']}) | {result_inet['train_time_s']/3600:.1f}h")
    print(f"Razem:          {results['summary']['total_time_s']/3600:.1f}h")
    print(f"Wyniki zapisane do: long_training_vit_best_150ep.json")
    print(f"{'='*65}")
