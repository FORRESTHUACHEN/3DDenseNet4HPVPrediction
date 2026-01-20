import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score,recall_score
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_curve

# ==================== Data Loading ====================
class MultiModalityMRIDataset(Dataset):
    """Dataset for loading multi-modality MRI scans"""
    
    def __init__(self, data_root, sample_ids, transform=None):
        self.data_root = data_root
        self.sample_ids = sample_ids
        self.transform = transform
        self.modalities = ['image', 'mask']
        
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        sample_path = os.path.join(self.data_root, sample_id)
        
        # Load all modalities
        modality_data = []
        for mod in self.modalities:
            file_path = os.path.join(sample_path, f'{mod}.nii.gz')
            nii_img = nib.load(file_path)
            img_data = nii_img.get_fdata()
            
            # Resize to 128x128x64 if needed
            if img_data.shape != (128, 128, 64):
                img_data = self._resize_volume(img_data, (128, 128, 64))
            
            # Normalize
            img_data = self._normalize(img_data)
            modality_data.append(img_data)
        
        # Stack modalities as channels (4, 128, 128, 64)
        volume = np.stack(modality_data, axis=0)
        
        # Load label
        label_path = os.path.join(sample_path, 'label.txt')
        with open(label_path, 'r') as f:
            label = int(f.read().strip())
        
        # Convert to tensors
        volume = torch.from_numpy(volume).float()
        label = torch.tensor(label, dtype=torch.long)
        
        if self.transform:
            volume = self.transform(volume)
        
        return volume, label
    
    def _resize_volume(self, volume, target_shape):
        """Resize volume using interpolation"""
        volume_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).float()
        resized = F.interpolate(volume_tensor, size=target_shape, mode='trilinear', align_corners=False)
        return resized.squeeze().numpy()
    
    def _normalize(self, volume):
        """Normalize volume to zero mean and unit variance"""
        mean = np.mean(volume)
        std = np.std(volume)
        if std > 0:
            volume = (volume - mean) / std
        return volume


# ==================== 3D DenseNet Architecture ====================
class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size):
        super(_DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, bn_size * growth_rate, kernel_size=1, bias=False)
        
        self.bn2 = nn.BatchNorm3d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
    
    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        return torch.cat([x, out], 1)


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, bn_size, growth_rate):
        super(_DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(_DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class _Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_Transition, self).__init__()
        self.bn = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)
    
    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        return self.pool(out)


class DenseNet3D(nn.Module):
    def __init__(self, in_channels=2, num_classes=2, growth_rate=32, 
                 block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4):
        super(DenseNet3D, self).__init__()
        
        # First convolution
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        
        # Dense blocks
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, num_features, bn_size, growth_rate)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features += num_layers * growth_rate
            
            if i != len(block_config) - 1:
                trans = _Transition(num_features, num_features // 2)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_features // 2
        
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Linear(num_features, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x):
        features = self.features(x)
        out = self.avgpool(features)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)


# ==================== Training and Evaluation ====================
class Trainer:
    def __init__(self, model, device, train_loader, val_loader, lr=1e-4, num_epochs=100):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
#        self.external_val_loader = external_val_loader
        self.num_epochs = num_epochs
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )
        
        self.best_auc = 0.0
        self.best_model_state = None
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        
        for volumes, labels in tqdm(self.train_loader, desc='Training'):
            volumes = volumes.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(volumes)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self, loader):
        self.model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for volumes, labels in tqdm(loader, desc='Evaluating'):
                volumes = volumes.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(volumes)
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        recall= recall_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)
        fpr,tpr,thresholds = roc_curve(all_labels, all_probs)
        auprc = average_precision_score(all_labels, all_probs)
        
        return {
            'accuracy': accuracy,
            'recall':recall,
            'auc': auc,
            'auprc': auprc,
            'predictions': all_preds,
            'probabilities': all_probs,
            'labels': all_labels
        }
    
    def train(self):
        print("Starting training...")
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Evaluate
            val_results = self.evaluate(self.val_loader)
            
#            external_results = self.evaluate(self.external_val_loader)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Accuracy: {val_results['accuracy']:.4f}")
            print(f"Val Recall: {val_results['recall']:.4f}")
            print(f"Val AUC: {val_results['auc']:.4f}")
            print(f"Val AUPRC: {val_results['auprc']:.4f}")
            
            
            # print(f"External Accuracy: {external_results['accuracy']:.4f}")
            # print(f"External Recall: {external_results['recall']:.4f}")
            # print(f"External AUC: {external_results['auc']:.4f}")
            # print(f"External AUPRC: {external_results['auprc']:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_results['auc'])
            
            # Save best model
            if val_results['auc'] > self.best_auc:
                self.best_auc = val_results['auc']
                self.best_model_state = self.model.state_dict().copy()
                print(f"New best model saved! AUC: {self.best_auc:.4f}")
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return self.model


# ==================== Main Pipeline ====================
def prepare_data(data_root, batch_size=4, val_split=0.1, random_seed=9):
    """Prepare train and validation dataloaders"""
    
    # Get all sample IDs
    sample_ids = [d for d in os.listdir(data_root) 
                  if os.path.isdir(os.path.join(data_root, d))]
    
    # Split into train and validation
    train_ids, val_ids = train_test_split(
        sample_ids, test_size=val_split, random_state=random_seed
    )
    
    print(f"Total samples: {len(sample_ids)}")
    print(f"Training samples: {len(train_ids)}")
    print(f"Validation samples: {len(val_ids)}")
    
    # Create datasets
    train_dataset = MultiModalityMRIDataset(data_root, train_ids)
    val_dataset = MultiModalityMRIDataset(data_root, val_ids)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True
    )
    
    return train_loader, val_loader


def main():
    # Configuration
    DATA_ROOT = 'D:/DatasetFromTCIA/HandNeckDataset/End2EndTranining/'  # Change this to your data path
#    DATA_ROOT2 = 'D:/DatasetFromTCIA/HandNeckDataset/ExternalValidation2/'
    BATCH_SIZE = 2  # Adjust based on GPU memory
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 2  # Binary classification
    
    # Set device
    device = torch.device('cuda:0')
    print(f"Using device: {device}")
    
    # Prepare data
    train_loader, val_loader = prepare_data(
        DATA_ROOT, batch_size=BATCH_SIZE, val_split=0.1
    )
    
    # external_train_loader, external_val_loader = prepare_data(
    #     DATA_ROOT2, batch_size=BATCH_SIZE, val_split=0.05
    # )
    
    # Create model
    model = DenseNet3D(
        in_channels=2,  # Image, Mask
        num_classes=NUM_CLASSES,
        growth_rate=32,
        block_config=(6, 12, 24, 16),  # DenseNet-121 configuration
        num_init_features=64
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
#        external_val_loader = external_train_loader,
        lr=LEARNING_RATE,
        num_epochs=NUM_EPOCHS
    )
    
    trained_model = trainer.train()
    
    # Final evaluation
    print("\n" + "="*50)
    print("Final Evaluation on Validation Set")
    print("="*50)
    final_results = trainer.evaluate(val_loader)
    print(f"Final Accuracy: {final_results['accuracy']:.4f}")
    print(f"Final AUC: {final_results['auc']:.4f}")
    print(f"Final AUPRC: {final_results['auprc']:.4f}")
    
    # Save model
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'best_auc': trainer.best_auc,
        'final_results': final_results
    }, 'best_model_3d_densenet4.pth')
    print("\nModel saved to 'best_model_3d_densenet4.pth'")


if __name__ == '__main__':
    main()
