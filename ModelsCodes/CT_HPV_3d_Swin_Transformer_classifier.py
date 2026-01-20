# swin3d_classifier.py
import os
import glob
import random
from pathlib import Path
from typing import List, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score,recall_score
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
# 3D Swin-like building blocks (compact)
# -----------------------------
def window_partition(x, window_size):
    """
    x: (B, C, H, W, D)
    window_size: tuple (wh, ww, wd)
    returns windows: (num_windows*B, C, wh, ww, wd)
    """
    B, C, H, W, D = x.shape
    wh, ww, wd = window_size
    x = x.view(B, C, H//wh, wh, W//ww, ww, D//wd, wd)
    x = x.permute(0,2,4,6,1,3,5,7).contiguous()  # (B, nH, nW, nD, C, wh, ww, wd)
    windows = x.view(-1, C, wh, ww, wd)
    return windows

def window_reverse(windows, window_size, H, W, D):
    """
    windows: (num_windows*B, C, wh, ww, wd)
    returns x: (B, C, H, W, D)
    """
    wh, ww, wd = window_size
    B = int(windows.shape[0] / ((H//wh)*(W//ww)*(D//wd)))
    x = windows.view(B, H//wh, W//ww, D//wd, -1, wh, ww, wd)
    x = x.permute(0,4,1,5,2,6,3,7).contiguous()
    x = x.view(B, -1, H, W, D)
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

class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (wh, ww, wd)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # relative position bias table (small)
        wh, ww, wd = window_size
        coords = torch.stack(torch.meshgrid(
            torch.arange(wh), torch.arange(ww), torch.arange(wd), indexing='ij'
        ), dim=0)  # (3,wh,ww,wd)
        coords_flat = coords.reshape(3, -1)
        relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]  # (3, N, N)
        relative_coords = relative_coords.permute(1,2,0).contiguous()  # (N, N, 3)
        self.register_buffer("relative_coords", relative_coords)
        self.relative_position_index = None
        # small MLP to produce bias from relative coords
        self.bias_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, num_heads)
        )

    def forward(self, x):
        """
        x: (num_windows*B, N, C)  where N = wh*ww*wd
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]  # each: (B_, N, num_heads, head_dim)
        q = q.permute(0,2,1,3)  # (B_, num_heads, N, head_dim)
        k = k.permute(0,2,1,3)
        v = v.permute(0,2,1,3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B_, num_heads, N, N)

        # compute relative bias via bias_mlp on precomputed relative coordinates
        # relative_coords: (N,N,3)
        rel = self.relative_coords.float()  # (N,N,3)
        bias = self.bias_mlp(rel)  # (N,N,num_heads)
        bias = bias.permute(2,0,1).unsqueeze(0)  # (1, num_heads, N, N)
        attn = attn + bias.to(attn.device)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v)  # (B_, num_heads, N, head_dim)
        out = out.permute(0,2,1,3).reshape(B_, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class SwinBlock3D(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=(4,4,4), shift_size=(0,0,0), mlp_ratio=4.0, drop=0., attn_drop=0., drop_path=0.0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # (H,W,D)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, window_size, num_heads, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = nn.Identity() if drop_path == 0. else nn.Dropout(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim*mlp_ratio), dropout=drop)

    def forward(self, x):
        """
        x: (B, H*W*D, C)
        """
        B, L, C = x.shape
        H, W, D = self.input_resolution
        assert L == H*W*D, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, D, C).permute(0,4,1,2,3)  # (B, C, H, W, D)

        # cyclic shift
        if any(s>0 for s in self.shift_size):
            shifted = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(2,3,4))
        else:
            shifted = x

        # partition windows
        wh, ww, wd = self.window_size
        Bc, Cc, Hc, Wc, Dc = shifted.shape
        # pad if not divisible
        pad_h = (wh - Hc % wh) % wh
        pad_w = (ww - Wc % ww) % ww
        pad_d = (wd - Dc % wd) % wd
        if pad_h or pad_w or pad_d:
            shifted = F.pad(shifted, (0,pad_d, 0,pad_w, 0,pad_h))  # pad last three dims
            Hc, Wc, Dc = shifted.shape[2], shifted.shape[3], shifted.shape[4]

        windows = window_partition(shifted, (wh, ww, wd))  # (num_windows*B, C, wh, ww, wd)
        nw = windows.shape[0]
        windows = windows.view(nw, Cc, -1).permute(0,2,1).contiguous()  # (nw, N, C)

        attn_windows = self.attn(windows)  # (nw, N, C)

        # merge windows
        attn_windows = attn_windows.permute(0,2,1).contiguous().view(nw, Cc, wh, ww, wd)
        shifted_back = window_reverse(attn_windows, (wh,ww,wd), Hc, Wc, Dc)  # (B, C, Hc, Wc, Dc)

        # remove padding
        if pad_h or pad_w or pad_d:
            shifted_back = shifted_back[:, :, :H, :W, :D]

        # reverse cyclic shift
        if any(s>0 for s in self.shift_size):
            x = torch.roll(shifted_back, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(2,3,4))
        else:
            x = shifted_back

        x = x.permute(0,2,3,4,1).contiguous().view(B, H*W*D, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed3D(nn.Module):
    def __init__(self, in_chans=2, embed_dim=96, patch_size=(4,4,4)):
        super().__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, C, H, W, D)
        x = self.proj(x)  # (B, embed_dim, H/ps, W/ps, D/ps)
        B, C, H, W, D = x.shape
        x = x.view(B, C, -1).permute(0,2,1).contiguous()  # (B, L, C)
        x = self.norm(x)
        return x, (H, W, D)

# -----------------------------
# SwinTransformer3D (compact stack)
# -----------------------------
class SimpleSwin3D(nn.Module):
    def __init__(self, in_chans=2, embed_dim=64, depths=[2,2], num_heads=[2,4], window_size=(4,4,4), num_classes=2):
        super().__init__()
        self.patch_embed = PatchEmbed3D(in_chans=in_chans, embed_dim=embed_dim, patch_size=window_size)
        # stack layers of SwinBlocks
        self.layers = nn.ModuleList()
        dim = embed_dim
        for i, d in enumerate(depths):
            blocks = []
            for j in range(d):
                shift = tuple( (ws//2 if (j%2)==1 else 0) for ws in window_size )
                blocks.append(SwinBlock3D(dim=dim, input_resolution=None, num_heads=num_heads[i],
                                          window_size=window_size, shift_size=shift))
            self.layers.append(nn.ModuleList(blocks))
            # downsample stage between layers (simple conv)
            if i < len(depths)-1:
                self.layers.append(nn.Conv3d(dim, dim*2, kernel_size=2, stride=2))
                dim = dim*2

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        # x: (B, C, H, W, D)
        x, (H, W, D) = self.patch_embed(x)  # (B, L, C)
        B, L, C = x.shape
        # set input_resolution for blocks
        pH, pW, pD = H, W, D
        for layer in self.layers:
            if isinstance(layer, nn.ModuleList):
                for blk in layer:
                    blk.input_resolution = (pH, pW, pD)
                    x = blk(x)
            else:
                # downsample conv
                # convert to (B,C,H,W,D)
                x3 = x.permute(0,2,1).contiguous().view(B, C, pH, pW, pD)
                x3 = layer(x3)
                B, C, pH, pW, pD = x3.shape
                x = x3.view(B, C, -1).permute(0,2,1).contiguous()
        # global pooling
        x = x.mean(dim=1)  # (B, C)
        out = self.head(x)
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
    random.seed(347)
    random.shuffle(idx)
    val_frac = 0.1
    n_val = int(n * val_frac)
    train_idx = idx[n_val:]
    val_idx = idx[:n_val]

    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(MultiModalMRIDataset(root_dir, target_shape=(128,128,64), augment=False), val_idx)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=2, pin_memory=True)

    # model
    model = SimpleSwin3D(in_chans=2, embed_dim=64, depths=[2,2], num_heads=[2,4], window_size=(4,4,4), num_classes=2)
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
                       'best_swin3d_checkpoint_v2.pt')
            print("Saved best checkpoint v2")

    # final evaluation report
    print("Final metrics:", metrics)

if __name__ == "__main__":
    main()
