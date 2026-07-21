import os
import tarfile
import urllib.request
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def download_imagenette(path="./data"):
    """Pobiera Imagenette2-320 (10-klasowy podzbiór ImageNet)."""
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    tgz_path = os.path.join(path, "imagenette2-320.tgz")
    extract_path = os.path.join(path, "imagenette2-320")

    if os.path.exists(extract_path):
        print("Imagenette2-320 już pobrane.")
        return extract_path

    os.makedirs(path, exist_ok=True)
    if not os.path.exists(tgz_path):
        print("Pobieranie Imagenette2-320 (~1.5GB)...")
        urllib.request.urlretrieve(url, tgz_path)
        print("Pobieranie zakończone.")

    print("Rozpakowywanie...")
    with tarfile.open(tgz_path, 'r:gz') as tar:
        tar.extractall(path)
    print("Rozpakowano.")

    os.remove(tgz_path)
    return extract_path


def data_loader_imagenet(path="./data", batch_size=128, img_size=32):
    """
    Ładuje Imagenette2-320 przeskalowane do img_size x img_size.
    Domyślnie 32x32 dla identycznych architektur co CIFAR-10.
    """
    dataset_path = download_imagenette(path)

    # Normalizacja ImageNet
    mu = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mu, std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mu, std)
    ])

    train_dataset = datasets.ImageFolder(
        os.path.join(dataset_path, 'train'),
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        os.path.join(dataset_path, 'val'),
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader
