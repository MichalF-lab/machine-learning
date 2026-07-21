import torch
import torch.nn as nn
from Vision_Transformer import VisionTransformer
from data_loader_imagenet import data_loader_imagenet
import time
import json
from cnn_model import count_parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

IMG_SIZE = 112
PATCH_SIZE = 8   # (112/8)^2 = 196 tokenow — jak oryginalny ViT paper (224/16=196)
EPOCHS = 150


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


if __name__ == "__main__":
    print(f"\n{'='*65}")
    print(f"=== ViT 150 epok — Imagenette2-320 @ {IMG_SIZE}x{IMG_SIZE} ===")
    print(f"patch_size={PATCH_SIZE} → ({IMG_SIZE}/{PATCH_SIZE})² = {(IMG_SIZE//PATCH_SIZE)**2} tokenow")
    print(f"{'='*65}\n")

    train_loader, test_loader = data_loader_imagenet(
        './data',
        batch_size=128,
        img_size=IMG_SIZE
    )
    print(f"Train: {len(train_loader.dataset)} obrazow | Test: {len(test_loader.dataset)} obrazow")

    model = VisionTransformer(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=256,
        num_heads=8,
        Nx=6,
        mlp_dim=1024,
        n_class=10,
        dropout=0.2
    ).to(device)

    num_params = count_parameters(model)
    print(f"Parametry: {num_params:,}\n")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    history = []
    start_time = time.time()

    for epoch in range(EPOCHS):
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

        if (epoch + 1) % 5 == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (epoch + 1)) * (EPOCHS - epoch - 1)
            history.append({
                "epoch": epoch + 1,
                "loss": running_loss / len(train_loader),
                "train_acc": train_acc,
                "test_acc": test_acc
            })
            print(f"  Epoka {epoch+1:3d}/150 | Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | ETA: {eta/3600:.2f}h", flush=True)

    train_time = time.time() - start_time
    best = max(history, key=lambda x: x['test_acc'])

    result = {
        "model": "ViT",
        "dataset": "Imagenette2-320",
        "img_size": IMG_SIZE,
        "patch_size": PATCH_SIZE,
        "num_tokens": (IMG_SIZE // PATCH_SIZE) ** 2,
        "config": {"embed_dim": 256, "num_heads": 8, "Nx": 6, "mlp_dim": 1024,
                   "dropout": 0.2, "optimizer": "AdamW", "weight_decay": 0.05,
                   "scheduler": "CosineAnnealingLR", "label_smoothing": 0.1},
        "num_params": num_params,
        "epochs": EPOCHS,
        "best_test_acc": best['test_acc'],
        "best_epoch": best['epoch'],
        "final_test_acc": history[-1]['test_acc'],
        "final_train_acc": history[-1]['train_acc'],
        "train_time_s": train_time,
        "time_per_epoch": train_time / EPOCHS,
        "history": history,
        "vs_32x32_150ep": 70.11,
        "vs_32x32_10ep": 66.52
    }

    with open("imagenet_112x112.json", "w", encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"\n{'='*65}")
    print(f"WYNIK KOŃCOWY:  {result['final_test_acc']:.2f}% (epoka 150)")
    print(f"NAJLEPSZY:      {result['best_test_acc']:.2f}% (epoka {result['best_epoch']})")
    print(f"vs 32x32 150ep: 70.11%  →  {result['best_test_acc'] - 70.11:+.2f}%")
    print(f"Czas:           {train_time/3600:.1f}h  ({result['time_per_epoch']:.1f}s/epoka)")
    print(f"Wyniki zapisane do: imagenet_112x112.json")
    print(f"{'='*65}")
