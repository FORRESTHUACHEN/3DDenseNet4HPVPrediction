# vit3d_classifier.py
import os
import glob
import random
from pathlib import Path
from typing import List, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, recall_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Utils: load NIfTI and preprocess
# -----------------------------
def load_nifti(path: str) -> np.ndarray:
    img = nib.load(path)
    data = img.get_fdata()
    return data.astype(np.float32)

def read_label(path: str) -> int:
    # label.txt contains single integer 0 or 1
    with open(path, 'r') as f:
        s = f.read().strip()
    return int(s)

def resample_to_shape(img: np.ndarray, target_shape: Tuple[int,int,int]) -> np.ndarray:
    # img shape: (H,W,D)
    zoom_factors = [t / s for s, t in zip(img.shape, target_shape)]
    return zoom(img, zoom_factors, order=1)  # linear

def normalize_zero_one(x: np.ndarray, eps=1e-8) -> np.ndarray:
    mn, mx = x.min(), x.max()
    if mx - mn < eps:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

# -----------------------------
# Dataset
# -----------------------------
class MultiModalMRIDataset(Dataset):
    def __init__(self, root_dir: str, target_shape=(128,128,64), modalities=('image','mask'),
                 augment=False):
        self.root_dir = Path(root_dir)
        self.modalities = modalities
        self.target_shape = target_shape
        self.augment = augment
        # each sample folder must contain flair.nii.gz, t1.nii.gz, t1ce.nii.gz, t2.nii.gz and label.txt
        self.samples = [p for p in self.root_dir.iterdir() if p.is_dir()]
        self.samples = sorted(self.samples)

    def __len__(self):
        return len(self.samples)

    def _load_sample(self, sample_dir: Path):
        channels = []
        for m in self.modalities:
            fpath = sample_dir / f"{m}.nii.gz"
            if not fpath.exists():
                raise FileNotFoundError(f"{fpath} not found")
            img = load_nifti(str(fpath))
            img = resample_to_shape(img, self.target_shape)
            img = normalize_zero_one(img)
            channels.append(img)
        x = np.stack(channels, axis=0)  # (C, H, W, D)
        label = read_label(str(sample_dir / "label.txt"))
        return x.astype(np.float32), int(label)

    def __getitem__(self, idx):
        x, y = self._load_sample(self.samples[idx])

        # simple augmentations
        if self.augment:
            # random flip in spatial dims
            if random.random() < 0.5:
                x = x[:, ::-1, :, :]
            if random.random() < 0.5:
                x = x[:, :, ::-1, :]
            if random.random() < 0.5:
                x = x[:, :, :, ::-1]
            # slight intensity jitter
            x = x + np.random.normal(0, 0.01, size=x.shape).astype(np.float32)
            x = np.clip(x, 0.0, 1.0)

        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.long)

# -----------------------------
# 3D ViT building blocks
# -----------------------------
class PatchEmbed3D(nn.Module):
    """3D Image to Patch Embedding"""
    def __init__(self, img_size=(128, 128, 64), patch_size=(16, 16, 16), in_chans=2, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])
        
        self.proj = nn.Conv3d(in_chans, embed_dim, 
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W, D = x.shape
        x = self.proj(x)  # (B, embed_dim, H', W', D')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

class Attention3D(nn.Module):
    """3D Multi-head Self Attention"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block3D(nn.Module):
    """3D Transformer Block"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention3D(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                                attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), dropout=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# -----------------------------
# 3D Vision Transformer
# -----------------------------
class SimpleViT3D(nn.Module):
    def __init__(self, img_size=(128, 128, 64), patch_size=(16, 16, 16), in_chans=2, 
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., 
                 qkv_bias=True, dropout=0., attn_dropout=0., num_classes=2):
        super().__init__()
        
        self.patch_embed = PatchEmbed3D(img_size=img_size, patch_size=patch_size,
                                        in_chans=in_chans, embed_dim=embed_dim)
        
        num_patches = self.patch_embed.num_patches
        
        # CLS token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block3D(embed_dim, num_heads, mlp_ratio, qkv_bias, dropout, attn_dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        # Classification head
        x = self.norm(x)
        cls_token = x[:, 0]  # Use CLS token for classification
        out = self.head(cls_token)
        
        return out

# -----------------------------
# Training and evaluation
# -----------------------------
def train_one_epoch(model, loader, optim, criterion, device):
    model.train()
    running_loss = 0.0
    for xb, yb in tqdm(loader, desc="train", leave=False):
        xb = xb.to(device)  # (B,C,H,W,D)
        yb = yb.to(device)
        optim.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optim.step()
        running_loss += float(loss.item()) * xb.size(0)
    return running_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    ys = []
    preds = []
    probs = []
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="eval", leave=False):
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            prob = F.softmax(logits, dim=1)[:,1].detach().cpu().numpy()
            pred = logits.argmax(dim=1).detach().cpu().numpy()
            ys.extend(yb.cpu().numpy().tolist())
            preds.extend(pred.tolist())
            probs.extend(prob.tolist())

    ys = np.array(ys)
    preds = np.array(preds)
    probs = np.array(probs)

    # metrics
    try:
        auc = roc_auc_score(ys, probs)
    except Exception:
        auc = float('nan')
    acc = accuracy_score(ys, preds)
    recall = recall_score(ys, preds)
    try:
        auprc = average_precision_score(ys, probs)
    except Exception:
        auprc = float('nan')

    return {'auc': auc, 'accuracy': acc,'recall':recall, 'auprc': auprc, 'preds': preds, 'probs': probs, 'labels': ys}

# -----------------------------
# Example training script
# -----------------------------
def main():
    root_dir = "D:/DatasetFromTCIA/HandNeckDataset/End2EndTranining/"  # <-- set this
    device = torch.device('cuda:1')

    # dataset
    dataset = MultiModalMRIDataset(root_dir, target_shape=(128,128,64), augment=True)
    # split
    n = len(dataset)
    idx = list(range(n))
    random.seed(456)
    random.shuffle(idx)
    val_frac = 0.1
    n_val = int(n * val_frac)
    train_idx = idx[n_val:]
    val_idx = idx[:n_val]

    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(MultiModalMRIDataset(root_dir, target_shape=(128,128,64), augment=False), val_idx)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=2, pin_memory=True)

    # model - using 3D ViT instead of Swin
    model = SimpleViT3D(
        img_size=(128, 128, 64),
        patch_size=(16, 16, 16),
        in_chans=2,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        dropout=0.1,
        attn_dropout=0.1,
        num_classes=2
    )
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_auc = 0.0
    epochs = 100
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train loss: {train_loss:.4f}")
        metrics = evaluate(model, val_loader, device)
        print(f"Val AUC: {metrics['auc']:.4f}  Acc: {metrics['accuracy']:.4f} Recall:{metrics['recall']:.4f} AUPRC: {metrics['auprc']:.4f}")

        # save best
        if not np.isnan(metrics['auc']) and metrics['auc'] > best_auc:
            best_auc = metrics['auc']
            torch.save({'epoch': epoch+1, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()},
                       'best_vit3d_checkpoint_v2.pt')
            print("Saved best checkpoint")

    # final evaluation report
    print("Final metrics:", metrics)

if __name__ == "__main__":
    main()