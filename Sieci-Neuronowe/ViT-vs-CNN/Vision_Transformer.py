import torch
from torch import nn
from patch_embedding import PatchEmbedding
from data_loader import data_loader
import time
from tqdm import tqdm


class Attention(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.):
        super().__init__()
        self.att = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True  # (B, N, D)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.att(x, x, x)
        return self.dropout(attn_output)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Sequential):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        return x + res


class PositionalEncoding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        pe = torch.zeros(num_patches + 1, embed_dim)
        position = torch.arange(0, num_patches + 1, 1).unsqueeze(1)  # (N, 1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, N+1, D)
        self.register_buffer("pos_embedding", pe)

    def forward(self, x):
        return x + self.pos_embedding


class VisionTransformer_block(nn.Sequential):
    def __init__(self, dim, n_heads, mlp_dim, dropout=0.1):
        super().__init__(
            ResidualAdd(
                PreNorm(dim, Attention(dim, n_heads, dropout))
            ),
            ResidualAdd(
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            )
        )


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        embed_dim=128,
        num_heads=8,
        Nx=6,
        mlp_dim=512,
        n_class=10,
        dropout=0.1
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim
        )
        self.num_patches = (img_size // patch_size) ** 2
        self.cls_tokens = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_encoding = PositionalEncoding(self.num_patches, embed_dim)
        self.transformer = nn.Sequential(
            *[VisionTransformer_block(embed_dim, num_heads, mlp_dim, dropout) for _ in range(Nx)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_class)

    def forward(self, x):
        x = self.patch_embed(x)
        B = x.shape[0]
        cls_tokens = self.cls_tokens.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        cls = x[:, 0]
        cls = self.norm(cls)
        cls = self.head(cls)
        return cls


if __name__ == "__main__":
    train_loader, test_loader = data_loader("./data")

    model = VisionTransformer(
        img_size=32,
        patch_size=4,
        embed_dim=128,
        num_heads=8,
        Nx=6,
        mlp_dim=512,
        n_class=10,
        dropout=0.3
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)

    num_epochs = 50
    num_batches = len(train_loader)
    total_batches = num_epochs * num_batches
    training_start = time.time()
    global_batch_idx = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        print(f"\nEpoka {epoch+1}/{num_epochs}")
        pbar = tqdm(train_loader, total=num_batches)

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = logits.max(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            global_batch_idx += 1
            elapsed_total = time.time() - training_start
            batches_left = total_batches - global_batch_idx
            eta_total = (elapsed_total / global_batch_idx) * batches_left

            pbar.set_description(
                f"loss: {running_loss/(batch_idx+1):.4f}, "
                f"acc: {100.*correct/total:.2f}%, "
                f"ETA: {eta_total:.0f}s"
            )

        print(
            f"Epoka {epoch+1}: "
            f"loss {running_loss / num_batches:.4f}, "
            f"acc {100. * correct / total:.2f}%"
        )

        if (epoch + 1) % 10 == 0:
            model.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    logits = model(images)
                    _, preds = logits.max(1)
                    test_total += labels.size(0)
                    test_correct += (preds == labels).sum().item()
            print(f">>> Test accuracy po epoce {epoch+1}: {100. * test_correct / test_total:.2f}%")
            model.train()

    total_time = time.time() - training_start
    print(f"\nCałkowity czas treningu: {total_time:.1f}s")

    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            _, preds = logits.max(1)
            test_total += labels.size(0)
            test_correct += (preds == labels).sum().item()
    print(f"Test accuracy: {100. * test_correct / test_total:.2f}%")
