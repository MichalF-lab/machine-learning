import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
import os
import cv2
import numpy as np
from pathlib import Path
import gc
from torch.utils.data import random_split


BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 7
N_FRAMES = 16

import torchvision.models as models

class CNNLSTMModel(nn.Module):
    def __init__(self, hidden_size=1024, num_layers=3):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        self.lstm = nn.LSTM(
            input_size=2048, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.15
        )
        
        self.fc = nn.Linear(hidden_size, 101)

    def forward(self, x):
            batch, frames, H, W, C = x.size()  # [batch, frames, H, W, C]
            x = x.view(batch * frames, H, W, C).permute(0, 3, 1, 2).contiguous()  # [B*F, C, H, W]
            x = torch.nn.functional.interpolate(x.float(), size=(112, 112), mode='bilinear')
            x = x / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
            x = (x - mean) / std
        
            # [B*F, 3, 112, 112]
            x = self.feature_extractor(x) # [B*F, 512, 1, 1]
            x = x.view(batch, frames, -1) # [B, F, 512]
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

def video_loader(path, n_frames=16, size=(112, 112)):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    frames = []
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
        frames.append(frame)
    cap.release()
    
    if len(frames) == 0: return None
    while len(frames) < n_frames:
        frames.append(frames[-1])
    return np.array(frames, dtype=np.uint8) # (N, H, W, C)

def collate_fn(batch):
    videos = [torch.from_numpy(b[0]) for b in batch if b[0] is not None]
    labels = [b[1] for b in batch if b[0] is not None]
    return torch.stack(videos), torch.tensor(labels)

class LazyVideoDataset(Dataset):
    def __init__(self, root, n_frames=16):
        self.root = Path(root)
        self.n_frames = n_frames
        self.extensions = ('.mp4', '.avi', '.mov', '.mkv')
        self.samples = []
        self.class_to_idx = {}
        for idx, class_dir in enumerate(sorted(self.root.iterdir())):
            if not class_dir.is_dir():
                continue
            self.class_to_idx[class_dir.name] = idx  
            for video_path in class_dir.iterdir():
                if video_path.suffix.lower() in self.extensions:
                    self.samples.append((str(video_path), idx))
        print(f"Found {len(self.samples)} videos in {len(self.class_to_idx)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        max_retries = 3
        for _ in range(max_retries):
            video = video_loader(path, n_frames=self.n_frames)
            if video is not None:
                return video, label
            idx = np.random.randint(0, len(self.samples))
            path, label = self.samples[idx]
        
        return np.zeros((self.n_frames, 112, 112, 3), dtype=np.uint8), label


if __name__ == '__main__': 
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    full_dataset = LazyVideoDataset(root='data/UCF-101', n_frames=N_FRAMES)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=8, pin_memory=True, collate_fn=collate_fn)

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                             num_workers=8, pin_memory=True, collate_fn=collate_fn)



    dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=6,  
        pin_memory=True,
        collate_fn=collate_fn,
        prefetch_factor=4,  
        persistent_workers=True,  
        timeout=60, 
    )

    model = CNNLSTMModel().to(DEVICE)
    

    for i, child in enumerate(model.feature_extractor.children()):
        if i < 4:
            for param in child.parameters():
                param.requires_grad = False

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=LEARNING_RATE
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        num_batches = 0
        
        for batch_idx, (video, labels) in enumerate(dataloader):
            if video is None:
                continue
                
            video = video.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(video)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            total_loss += loss.item()
            num_batches += 1
            
           
            if batch_idx % 20 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
            # garbage collection
            if batch_idx % 300 == 0:
                gc.collect()
        
        model.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0
        for video, labels in test_loader:
                if video is None: continue
                
                video, labels = video.to(DEVICE), labels.to(DEVICE)
                outputs = model(video)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        avg_test_acc = 100. * test_correct / test_total if test_total > 0 else 0
        print(f"Test Loss: {test_loss/len(test_loader):.4f}, Test Acc: {avg_test_acc:.2f}%")
        

    print("Training completed successfully")