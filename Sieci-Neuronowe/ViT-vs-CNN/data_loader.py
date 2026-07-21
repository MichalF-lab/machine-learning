from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def data_loader(path="./data", batch_size=128):
    mu = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mu, std)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mu, std)
    ])

    cifar10_train = datasets.CIFAR10(
        root=path,
        train=True,
        download=True,
        transform=train_transform
    )

    cifar10_test = datasets.CIFAR10(
        root=path,
        train=False,
        download=True,
        transform=test_transform
    )

    train_loader = DataLoader(
        cifar10_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=3,
        pin_memory=True
    )

    test_loader = DataLoader(
        cifar10_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=3,
        pin_memory=True
    )

    return train_loader, test_loader
