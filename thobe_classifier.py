# =============================================================
#  🇵🇸 Palestinian Thobe Regional Classifier
#  Transfer Learning: EfficientNetB0 & MobileNetV2
#  ── PyTorch Implementation ──
# =============================================================
#
#  Dataset structure expected:
#  Dataset/
#  ├── nablus/
#  ├── bethlehem/
#  └── jaffa/
#
#  The script auto-splits into train / val / test (80/10/10)
# =============================================================

import os
import time
import random
import shutil
import warnings
import copy

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    mobilenet_v2,   MobileNet_V2_Weights,
)
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score
)

from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore')

# =============================================================
#  CONSTANTS
# =============================================================

IMG_SIZE    = 224
BATCH_SIZE  = 8          # Smaller batch → more gradient updates per epoch (important with ~140 train images)
CLASSES     = ['nablus', 'bethlehem', 'jaffa']
NUM_CLASSES = len(CLASSES)
SEED        = 42
DATASET_DIR = 'Dataset'
DATA_DIR    = 'data_split'

# ── Tiny-dataset knobs ──────────────────────────────────────
# Total images ≈ 179  →  train ≈ 143, val ≈ 18, test ≈ 18
# Strategy: heavy augmentation + label smoothing + WeightedRandomSampler
LABEL_SMOOTHING = 0.1    # Prevents overconfident predictions on tiny data
LR_HEAD         = 1e-3   # LR for the custom classification head
LR_FINETUNE     = 5e-5   # Lower LR when unfreezing backbone layers

# Device
DEVICE = (
    torch.device('cuda')  if torch.cuda.is_available() else
    torch.device('mps')   if torch.backends.mps.is_available() else
    torch.device('cpu')
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == 'cuda':
    torch.cuda.manual_seed_all(SEED)

print('=' * 60)
print('  🇵🇸 Palestinian Thobe Classifier  (PyTorch)')
print('  EfficientNetB0 & MobileNetV2')
print('=' * 60)
print(f'PyTorch : {torch.__version__}')
print(f'Device  : {DEVICE}')
print()


# =============================================================
#  HELPER: AUTO SPLIT DATASET
# =============================================================

def prepare_dataset(src_dir, dst_dir=DATA_DIR,
                    train_ratio=0.80, val_ratio=0.10, seed=SEED):
    """Split flat Dataset/ into train / val / test. Skips if done."""
    if os.path.exists(dst_dir):
        print(f'✅  {dst_dir}/ already exists — skipping split.\n')
        return dst_dir

    classes = [d for d in os.listdir(src_dir)
               if os.path.isdir(os.path.join(src_dir, d))]
    print(f'Found classes : {classes}')
    print(f'Split ratio   : train={train_ratio} | val={val_ratio} '
          f'| test={1-train_ratio-val_ratio:.2f}\n')

    for cls in classes:
        imgs = os.listdir(os.path.join(src_dir, cls))
        random.seed(seed)
        random.shuffle(imgs)

        n_train = int(len(imgs) * train_ratio)
        n_val   = int(len(imgs) * val_ratio)

        split_map = {
            'train': imgs[:n_train],
            'val'  : imgs[n_train:n_train + n_val],
            'test' : imgs[n_train + n_val:]
        }

        for split, files in split_map.items():
            out = os.path.join(dst_dir, split, cls)
            os.makedirs(out, exist_ok=True)
            for f in files:
                shutil.copy(
                    os.path.join(src_dir, cls, f),
                    os.path.join(out, f)
                )
            print(f'  {split:5s}/{cls:12s}  →  {len(files)} images')

    print('\n✅  Dataset split complete!\n')
    return dst_dir


# =============================================================
#  EDA
# =============================================================

def run_eda(data_dir, classes=CLASSES):
    splits = ['train', 'val', 'test']
    counts = {cls: {} for cls in classes}

    print('=' * 52)
    print('📊  DATASET OVERVIEW')
    print('=' * 52)
    print(f"{'Class':<14} {'Train':>6} {'Val':>6} {'Test':>6} {'Total':>7}")
    print('-' * 52)
    for cls in classes:
        for split in splits:
            path = os.path.join(data_dir, split, cls)
            counts[cls][split] = len(os.listdir(path)) if os.path.exists(path) else 0
        total = sum(counts[cls].values())
        print(f"{cls:<14} {counts[cls]['train']:>6} "
              f"{counts[cls]['val']:>6} {counts[cls]['test']:>6} {total:>7}")
    print('=' * 52)

    # Distribution bar chart
    x      = np.arange(len(classes))
    width  = 0.25
    colors = ['#2196F3', '#FF9800', '#4CAF50']

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, split in enumerate(splits):
        vals = [counts[cls][split] for cls in classes]
        bars = ax.bar(x + (i - 1) * width, vals, width,
                      label=split.capitalize(), color=colors[i], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3, str(v),
                    ha='center', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in classes], fontsize=12)
    ax.set_ylabel('Number of Images', fontsize=11)
    ax.set_title('🇵🇸 Class Distribution Across Splits', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('eda_distribution.png', dpi=150)
    plt.close()
    print('  Saved: eda_distribution.png')

    # Sample images
    class_colors = {'bethlehem': '#e53935', 'nablus': '#37474f', 'jaffa': '#f9a825'}
    fig, axes = plt.subplots(3, 4, figsize=(14, 11))
    fig.suptitle('🇵🇸 Sample Thobes per Region', fontsize=16, fontweight='bold')
    for i, cls in enumerate(classes):
        cls_path = os.path.join(data_dir, 'train', cls)
        imgs     = random.sample(os.listdir(cls_path),
                                 min(4, len(os.listdir(cls_path))))
        for j, img_name in enumerate(imgs):
            img = Image.open(os.path.join(cls_path, img_name))
            axes[i, j].imshow(img)
            axes[i, j].set_title(cls.capitalize(), fontsize=11,
                                  color=class_colors.get(cls, 'black'),
                                  fontweight='bold')
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.savefig('eda_samples.png', dpi=150)
    plt.close()
    print('  Saved: eda_samples.png')

    # Size distribution
    heights, widths = [], []
    for cls in classes:
        cls_path = os.path.join(data_dir, 'train', cls)
        for img_name in os.listdir(cls_path):
            img = Image.open(os.path.join(cls_path, img_name))
            w, h = img.size
            widths.append(w)
            heights.append(h)

    print(f'\n📐  Image sizes:')
    print(f'  Height → min:{min(heights)} max:{max(heights)} mean:{np.mean(heights):.0f}')
    print(f'  Width  → min:{min(widths)}  max:{max(widths)}  mean:{np.mean(widths):.0f}')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(heights, bins=15, color='#2196F3', edgecolor='black', alpha=0.8)
    axes[0].axvline(IMG_SIZE, color='red', linestyle='--', label=f'Target {IMG_SIZE}px')
    axes[0].set_title('Heights Distribution')
    axes[0].legend()
    axes[1].hist(widths, bins=15, color='#FF9800', edgecolor='black', alpha=0.8)
    axes[1].axvline(IMG_SIZE, color='red', linestyle='--', label=f'Target {IMG_SIZE}px')
    axes[1].set_title('Widths Distribution')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig('eda_sizes.png', dpi=150)
    plt.close()
    print('  Saved: eda_sizes.png\n')


# =============================================================
#  DATA TRANSFORMS & LOADERS
# =============================================================

# ImageNet normalization — used by both models in PyTorch
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def make_transforms():
    """
    Returns (train_transform, val_test_transform).

    ── Tiny-dataset augmentation strategy (~179 images total) ──
    Heavy augmentation acts as a regulariser and artificially
    expands the effective training set. Choices are thobe-aware:

    • RandomHorizontalFlip       – thobes are left-right symmetric
    • RandomRotation(±10°)       – slight tilt is realistic
    • RandomResizedCrop(0.75-1)  – zoom in/out on garment details
    • ColorJitter                – lighting & colour temperature variation
    • RandomGrayscale(p=0.05)    – rare but forces shape-not-colour learning
    • RandomPerspective           – simulates camera angle changes
    • RandomErasing              – occlusion regularisation (bag / arm overlay)

    Val/test: only resize + centre-crop + normalise (no augmentation).
    """
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),   # slightly larger
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.4),
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3,
            saturation=0.3, hue=0.05
        ),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        # RandomErasing: randomly masks 2-20% of the image → occlusion robustness
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.20),
                                  ratio=(0.3, 3.3), value='random'),
    ])

    val_test_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_tf, val_test_tf


def make_loaders(data_dir=DATA_DIR, batch_size=BATCH_SIZE):
    """
    Build DataLoaders for train / val / test.

    ── Tiny-dataset additions ──
    WeightedRandomSampler: compensates for class imbalance
    (bethlehem=69, nablus=56, jaffa=54). Each class gets equal
    expected representation per epoch regardless of count.
    """
    train_tf, val_test_tf = make_transforms()

    train_ds = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, 'val'),   transform=val_test_tf)
    test_ds  = datasets.ImageFolder(os.path.join(data_dir, 'test'),  transform=val_test_tf)

    print(f'ImageFolder class indices : {train_ds.class_to_idx}')

    # ── WeightedRandomSampler ─────────────────────────────
    # Weight each sample by the inverse frequency of its class
    class_counts  = np.bincount(train_ds.targets)
    class_weights = 1.0 / class_counts.astype(float)
    sample_weights = torch.tensor(
        [class_weights[t] for t in train_ds.targets], dtype=torch.float
    )
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    print(f'Class counts (train) : {dict(zip(train_ds.class_to_idx.keys(), class_counts))}')
    print(f'Sample weights       : {class_weights.round(4)}')

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler,          # replaces shuffle=True
                              num_workers=0,
                              pin_memory=(DEVICE.type == 'cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0,
                              pin_memory=(DEVICE.type == 'cuda'))
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=0,
                              pin_memory=(DEVICE.type == 'cuda'))

    return train_loader, val_loader, test_loader, train_ds.class_to_idx


# =============================================================
#  MODEL BUILDER
# =============================================================

def build_efficientnet(freeze=True, lr=LR_HEAD):
    """
    EfficientNetB0 with custom classification head.

    Tiny-dataset changes vs generic version:
    • Stronger Dropout (0.5 / 0.4) to fight overfitting
    • Label smoothing in CrossEntropyLoss (set in fit())
    """
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),           # ↑ was 0.4
        nn.Linear(256, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),           # ↑ was 0.3
        nn.Linear(128, NUM_CLASSES),
    )

    for param in model.classifier.parameters():
        param.requires_grad = True

    model = model.to(DEVICE)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    return model, optimizer


def build_mobilenet(freeze=True, lr=LR_HEAD):
    """
    MobileNetV2 with custom classification head.
    Same tiny-dataset tweaks as EfficientNetB0 above.
    """
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(256, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(128, NUM_CLASSES),
    )

    for param in model.classifier.parameters():
        param.requires_grad = True

    model = model.to(DEVICE)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    return model, optimizer


def param_summary(model, label):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\n── {label}')
    print(f'   Total params     : {total:>10,}')
    print(f'   Trainable params : {trainable:>10,}')
    print(f'   Frozen params    : {total - trainable:>10,}')


# =============================================================
#  TRAINING & EVALUATION LOOP
# =============================================================

def train_one_epoch(model, loader, criterion, optimizer, max_grad_norm=1.0):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        # Gradient clipping — prevents exploding gradients on tiny datasets.
        # Clips the global norm of all parameter gradients to max_grad_norm.
        # If the norm exceeds 1.0 the gradients are scaled down proportionally;
        # if it's already below 1.0 nothing changes.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct  += preds.eq(labels).sum().item()
        total    += images.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss    = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct  += preds.eq(labels).sum().item()
        total    += images.size(0)
    return running_loss / total, correct / total


def fit(model, optimizer, train_loader, val_loader,
        epochs=40, patience=15, save_path='model.pt',
        scheduler_patience=7):
    """
    Full training loop — tuned for tiny datasets (~140 train images).

    Changes vs generic version:
    • epochs=40, patience=15  — more room to converge slowly
    • label_smoothing=0.1     — prevents overconfident softmax on small data
    • ReduceLROnPlateau patience=7 — gives more time before halving LR
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5,
        patience=scheduler_patience, min_lr=1e-7
    )

    best_val_acc  = 0.0
    best_weights  = None
    patience_ctr  = 0
    history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'lr': []
}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)        
        vl_loss, vl_acc = evaluate(model, val_loader, criterion)
        scheduler.step(vl_loss)
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(vl_loss)
        history['val_acc'].append(vl_acc)

        print(f'  Epoch {epoch:>3d}/{epochs}  '
              f'loss: {tr_loss:.4f}  acc: {tr_acc:.4f}  '
              f'val_loss: {vl_loss:.4f}  val_acc: {vl_acc:.4f}  '
              f'lr: {optimizer.param_groups[0]["lr"]:.2e}')

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, save_path)
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f'  ⏹  Early stopping at epoch {epoch} '
                      f'(best val_acc={best_val_acc:.4f})')
                break

    model.load_state_dict(best_weights)
    return history


@torch.no_grad()
def get_predictions(model, loader):
    """Returns (y_true, y_pred) numpy arrays."""
    model.eval()
    all_true, all_pred = [], []
    for images, labels in loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, preds = outputs.max(1)
        all_true.extend(labels.cpu().numpy())
        all_pred.extend(preds.cpu().numpy())
    return np.array(all_true), np.array(all_pred)

@torch.no_grad()
def get_probabilities(model, loader):
    """
    Returns:
        y_true  -> true labels
        y_score -> predicted probabilities
    """
    model.eval()

    all_true = []
    all_scores = []

    for images, labels in loader:

        images = images.to(DEVICE)

        outputs = model(images)

        probs = torch.softmax(outputs, dim=1)

        all_true.extend(labels.cpu().numpy())
        all_scores.extend(probs.cpu().numpy())

    return np.array(all_true), np.array(all_scores)

@torch.no_grad()
def test_accuracy(model, loader):
    """Quick accuracy on test set."""
    criterion = nn.CrossEntropyLoss()
    _, acc = evaluate(model, loader, criterion)
    return acc


# =============================================================
#  PLOT HELPERS
# =============================================================

def plot_two_models(h_eff, h_mob, title):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'🇵🇸 {title}', fontsize=14, fontweight='bold')
    for ax, metric in zip(axes, ['acc', 'loss']):
        ax.plot(h_eff[f'val_{metric}'], label='EfficientNetB0',
                color='#2196F3', linewidth=2)
        ax.plot(h_mob[f'val_{metric}'], label='MobileNetV2',
                color='#FF9800', linewidth=2)
        ax.set_title(f'Validation {"Accuracy" if metric=="acc" else "Loss"}')
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)
        if metric == 'acc':
            ax.set_ylim(0, 1)
    plt.tight_layout()
    fname = f'plot_{title.replace(" ", "_").lower()}.png'
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f'  Saved: {fname}')

def plot_learning_rate(history, title):
    plt.figure(figsize=(8, 5))

    plt.plot(history['lr'], linewidth=2)

    plt.title(f'Learning Rate Schedule — {title}',
              fontsize=13, fontweight='bold')

    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')

    plt.yscale('log')   # very important for LR visualization

    plt.grid(True, alpha=0.3)

    fname = f'lr_{title.replace(" ", "_").lower()}.png'

    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

    print(f'  Saved: {fname}')

def plot_training_curves(history, title):
    """
    Creates ONE image with:
    Left  -> Train Loss + Validation Loss
    Right -> Train Accuracy + Validation Accuracy
    """

    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # =====================================================
    # LOSS CURVES
    # =====================================================
    axes[0].plot(
        epochs,
        history['train_loss'],
        label='Train Loss',
        linewidth=2
    )

    axes[0].plot(
        epochs,
        history['val_loss'],
        label='Validation Loss',
        linewidth=2
    )

    axes[0].set_title('Model Loss', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')

    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # =====================================================
    # ACCURACY CURVES
    # =====================================================
    axes[1].plot(
        epochs,
        history['train_acc'],
        label='Train Accuracy',
        linewidth=2
    )

    axes[1].plot(
        epochs,
        history['val_acc'],
        label='Validation Accuracy',
        linewidth=2
    )

    axes[1].set_title('Model Accuracy', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')

    axes[1].set_ylim(0, 1)

    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # =====================================================
    # SAVE
    # =====================================================
    fig.suptitle(
        f'🇵🇸 Training Curves — {title}',
        fontsize=15,
        fontweight='bold'
    )

    plt.tight_layout()

    fname = f'training_curves_{title.replace(" ", "_").lower()}.png'

    plt.savefig(fname, dpi=150)

    plt.close()

    print(f'  Saved: {fname}')

def plot_strategies(h_a, h_b, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'🇵🇸 {model_name} — Strategy A vs B',
                 fontsize=14, fontweight='bold')
    for ax, metric in zip(axes, ['acc', 'loss']):
        ax.plot(h_a[f'val_{metric}'], label='Strategy A (last N layers)',
                color='#2196F3', linewidth=2)
        ax.plot(h_b[f'val_{metric}'], label='Strategy B (full blocks)',
                color='#F44336', linewidth=2)
        ax.set_title(f'Validation {"Accuracy" if metric=="acc" else "Loss"}')
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)
        if metric == 'acc':
            ax.set_ylim(0, 1)
    plt.tight_layout()
    fname = f'strategies_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f'  Saved: {fname}')


def plot_confusion(model, loader, title, class_to_idx, color='Blues'):
    y_true, y_pred = get_predictions(model, loader)
    cm = confusion_matrix(y_true, y_pred)

    # Map idx → class name using class_to_idx from ImageFolder
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    tick_labels  = [idx_to_class[i].capitalize() for i in range(NUM_CLASSES)]

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap=color,
                xticklabels=tick_labels, yticklabels=tick_labels,
                linewidths=0.5)
    plt.title(f'🇵🇸 Confusion Matrix — {title}', fontsize=13, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    fname = f'cm_{title.replace(" ", "_").lower()}.png'
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f'  Saved: {fname}')

    print(f'\n📋  Classification Report — {title}:')
    print(classification_report(y_true, y_pred, target_names=tick_labels))

def plot_roc_curve(model, loader, title, class_to_idx):

    y_true, y_score = get_probabilities(model, loader)

    # Convert labels to one-hot
    y_true_bin = label_binarize(
        y_true,
        classes=np.arange(NUM_CLASSES)
    )

    idx_to_class = {v: k for k, v in class_to_idx.items()}

    plt.figure(figsize=(8, 6))

    for i in range(NUM_CLASSES):

        fpr, tpr, _ = roc_curve(
            y_true_bin[:, i],
            y_score[:, i]
        )

        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f'{idx_to_class[i].capitalize()} '
                  f'(AUC = {roc_auc:.2f})'
        )

    # Random classifier line
    plt.plot([0, 1], [0, 1], linestyle='--')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.title(f'ROC Curve — {title}',
              fontsize=13,
              fontweight='bold')

    plt.legend(loc='lower right')

    plt.grid(alpha=0.3)

    fname = f'roc_{title.replace(" ", "_").lower()}.png'

    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

    print(f'  Saved: {fname}')

def plot_final_comparison(val_results):
    labels   = ['Phase 1\n(frozen)', 'Strategy A\n(last N layers)',
                 'Strategy B\n(full blocks)']
    eff_accs = [val_results['eff_p1'], val_results['eff_a'], val_results['eff_b']]
    mob_accs = [val_results['mob_p1'], val_results['mob_a'], val_results['mob_b']]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - width / 2, [a * 100 for a in eff_accs], width,
                   label='EfficientNetB0', color='#2196F3', alpha=0.85)
    bars2 = ax.bar(x + width / 2, [a * 100 for a in mob_accs], width,
                   label='MobileNetV2',   color='#FF9800', alpha=0.85)

    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%',
                ha='center', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Val Accuracy (%)', fontsize=12)
    ax.set_title('🇵🇸 EfficientNetB0 vs MobileNetV2 — Strategy Comparison (Val Set)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 115)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('final_comparison.png', dpi=150)
    plt.close()
    print('  Saved: final_comparison.png')


# =============================================================
#  PART 1 — BASIC TRANSFER LEARNING (frozen base)
# =============================================================

def part1(train_loader, val_loader):
    """
    Phase 1: frozen base. Accuracy reported on val_loader —
    test_loader is NOT touched here. It is reserved for the
    single final evaluation after the best strategy is chosen.
    """
    print('\n' + '=' * 60)
    print('  PART 1 — Basic Transfer Learning (Frozen Base)')
    print('=' * 60)

    # ── EfficientNetB0 ────────────────────────────────────
    print('\n🚀  Training EfficientNetB0 (frozen)...')
    eff_model, eff_opt = build_efficientnet(freeze=True, lr=LR_HEAD)
    param_summary(eff_model, 'EfficientNetB0 Phase 1')

    eff_hist1 = fit(eff_model, eff_opt, train_loader, val_loader,
                    epochs=40, patience=15, save_path='eff_phase1.pt')
    eff_p1_acc = test_accuracy(eff_model, val_loader)   # val — for comparison only
    print(f'\n✅  EfficientNetB0 Phase 1 Val Accuracy: {eff_p1_acc*100:.2f}%')

    # ── MobileNetV2 ───────────────────────────────────────
    print('\n🚀  Training MobileNetV2 (frozen)...')
    mob_model, mob_opt = build_mobilenet(freeze=True, lr=LR_HEAD)
    param_summary(mob_model, 'MobileNetV2 Phase 1')

    mob_hist1 = fit(mob_model, mob_opt, train_loader, val_loader,
                    epochs=40, patience=15, save_path='mob_phase1.pt')
    mob_p1_acc = test_accuracy(mob_model, val_loader)   # val — for comparison only
    print(f'\n✅  MobileNetV2 Phase 1 Val Accuracy: {mob_p1_acc*100:.2f}%')

    plot_two_models(eff_hist1, mob_hist1, 'Part 1 — Frozen Base')
    plot_learning_rate(eff_hist1, 'EfficientNetB0 Phase 1')
    plot_learning_rate(mob_hist1, 'MobileNetV2 Phase 1')
    plot_training_curves(eff_hist1, 'EfficientNetB0 Phase 1')
    plot_training_curves(mob_hist1, 'MobileNetV2 Phase 1')

    return (eff_model, eff_hist1, eff_p1_acc,
            mob_model, mob_hist1, mob_p1_acc)


# =============================================================
#  PART 2 — UNFREEZING STRATEGIES
# =============================================================

def print_block_boundaries_eff():
    """Print EfficientNetB0 block boundaries."""
    model = efficientnet_b0(weights=None)
    print('\nEfficientNetB0 — Feature blocks:')
    for i, (name, module) in enumerate(model.features.named_children()):
        print(f'  features.{name:<4}  →  {type(module).__name__}')
    print(f'  Total feature blocks: {len(list(model.features.children()))}')
    del model


def print_block_boundaries_mob():
    """Print MobileNetV2 block boundaries."""
    model = mobilenet_v2(weights=None)
    print('\nMobileNetV2 — Feature blocks:')
    for i, (name, module) in enumerate(model.features.named_children()):
        print(f'  features.{name:<4}  →  {type(module).__name__}')
    print(f'  Total feature blocks: {len(list(model.features.children()))}')
    del model


def strategy_a_eff(train_loader, val_loader, n_layers=10):
    """
    Strategy A — EfficientNetB0: unfreeze the last N sub-modules
    of model.features.

    Accuracy is reported on val_loader so it can be fairly compared
    against Strategy B to pick the winner. test_loader is never
    seen here — it is reserved for the single final evaluation in main().
    """
    print(f'\n── Strategy A (EfficientNetB0) — Unfreeze last {n_layers} feature blocks')

    model, opt = build_efficientnet(freeze=True, lr=LR_HEAD)
    fit(model, opt, train_loader, val_loader,
        epochs=25, patience=12, save_path='eff_a_p1.pt')

    features_children = list(model.features.children())
    total_blocks      = len(features_children)
    unfreeze_idx      = max(0, total_blocks - n_layers)

    for i, block in enumerate(features_children):
        for param in block.parameters():
            param.requires_grad = (i >= unfreeze_idx)

    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'   Unfreeze from feature block index : {unfreeze_idx}')
    print(f'   Trainable params                  : {trainable:,}')

    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=LR_FINETUNE)
    hist = fit(model, opt, train_loader, val_loader,
               epochs=30, patience=15, save_path='eff_strategy_a.pt')

    val_acc = test_accuracy(model, val_loader)   # val — for strategy comparison only
    print(f'\n✅  EfficientNetB0 Strategy A Val Accuracy: {val_acc*100:.2f}%')
    return model, hist, val_acc


def strategy_a_mob(train_loader, val_loader, n_layers=10):
    """
    Strategy A — MobileNetV2: unfreeze the last N sub-modules of features.
    Accuracy on val_loader only — test_loader reserved for final eval in main().
    """
    print(f'\n── Strategy A (MobileNetV2) — Unfreeze last {n_layers} feature blocks')

    model, opt = build_mobilenet(freeze=True, lr=LR_HEAD)
    fit(model, opt, train_loader, val_loader,
        epochs=25, patience=12, save_path='mob_a_p1.pt')

    features_children = list(model.features.children())
    total_blocks      = len(features_children)
    unfreeze_idx      = max(0, total_blocks - n_layers)

    for i, block in enumerate(features_children):
        for param in block.parameters():
            param.requires_grad = (i >= unfreeze_idx)

    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'   Unfreeze from feature block index : {unfreeze_idx}')
    print(f'   Trainable params                  : {trainable:,}')

    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=LR_FINETUNE)
    hist = fit(model, opt, train_loader, val_loader,
               epochs=30, patience=15, save_path='mob_strategy_a.pt')

    val_acc = test_accuracy(model, val_loader)   # val — for strategy comparison only
    print(f'\n✅  MobileNetV2 Strategy A Val Accuracy: {val_acc*100:.2f}%')
    return model, hist, val_acc


def strategy_b_eff(train_loader, val_loader, unfreeze_from_block=6):
    """
    Strategy B — EfficientNetB0: unfreeze complete architectural blocks
    from index unfreeze_from_block onward (default: blocks 6, 7, 8).
    Accuracy on val_loader only — test_loader reserved for final eval in main().
    """
    print(f'\n── Strategy B (EfficientNetB0) — Unfreeze from block {unfreeze_from_block}')

    model, opt = build_efficientnet(freeze=True, lr=LR_HEAD)
    fit(model, opt, train_loader, val_loader,
        epochs=25, patience=12, save_path='eff_b_p1.pt')

    features_children = list(model.features.children())
    unfrozen_names    = []
    for i, block in enumerate(features_children):
        unfreeze = (i >= unfreeze_from_block)
        for param in block.parameters():
            param.requires_grad = unfreeze
        if unfreeze:
            unfrozen_names.append(f'features[{i}]')

    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'   Unfrozen blocks  : {unfrozen_names}')
    print(f'   Trainable params : {trainable:,}')

    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=LR_FINETUNE)
    hist = fit(model, opt, train_loader, val_loader,
               epochs=30, patience=15, save_path='eff_strategy_b.pt')

    val_acc = test_accuracy(model, val_loader)   # val — for strategy comparison only
    print(f'\n✅  EfficientNetB0 Strategy B Val Accuracy: {val_acc*100:.2f}%')
    return model, hist, val_acc


def strategy_b_mob(train_loader, val_loader, unfreeze_from_block=15):
    """
    Strategy B — MobileNetV2: unfreeze complete architectural blocks
    from index unfreeze_from_block onward (default: last 4 blocks).
    Accuracy on val_loader only — test_loader reserved for final eval in main().
    """
    print(f'\n── Strategy B (MobileNetV2) — Unfreeze from block {unfreeze_from_block}')

    model, opt = build_mobilenet(freeze=True, lr=LR_HEAD)
    fit(model, opt, train_loader, val_loader,
        epochs=25, patience=12, save_path='mob_b_p1.pt')

    features_children = list(model.features.children())
    unfrozen_names    = []
    for i, block in enumerate(features_children):
        unfreeze = (i >= unfreeze_from_block)
        for param in block.parameters():
            param.requires_grad = unfreeze
        if unfreeze:
            unfrozen_names.append(f'features[{i}]')

    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'   Unfrozen blocks  : {unfrozen_names}')
    print(f'   Trainable params : {trainable:,}')

    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=LR_FINETUNE)
    hist = fit(model, opt, train_loader, val_loader,
               epochs=30, patience=15, save_path='mob_strategy_b.pt')

    val_acc = test_accuracy(model, val_loader)   # val — for strategy comparison only
    print(f'\n✅  MobileNetV2 Strategy B Val Accuracy: {val_acc*100:.2f}%')
    return model, hist, val_acc


# =============================================================
#  PART 3 — ANALYSIS & COMPARISON
# =============================================================

def part3_analysis(val_results, test_results, best_eff, best_mob,
                   test_loader, class_to_idx):

    print('\n' + '=' * 60)
    print('  PART 3 — Analysis & Comparison')
    print('=' * 60)

    # Confusion matrices
    plot_confusion(best_eff, test_loader,
               'EfficientNetB0 Best', class_to_idx, 'Blues')

    plot_roc_curve(best_eff, test_loader,
               'EfficientNetB0 Best', class_to_idx)

    plot_confusion(best_mob, test_loader,
               'MobileNetV2 Best', class_to_idx, 'Greens')

    plot_roc_curve(best_mob, test_loader,
               'MobileNetV2 Best', class_to_idx)

    # Inference speed
    print('\n⚡  Measuring inference speed...')
    # Warm up
    get_predictions(best_eff, test_loader)
    get_predictions(best_mob, test_loader)

    t0 = time.time()
    for _ in range(3):
        get_predictions(best_eff, test_loader)
    eff_time = (time.time() - t0) / 3

    t0 = time.time()
    for _ in range(3):
        get_predictions(best_mob, test_loader)
    mob_time = (time.time() - t0) / 3

    n_test  = len(test_loader.dataset)
    faster  = 'MobileNetV2' if mob_time < eff_time else 'EfficientNetB0'
    ratio   = max(eff_time, mob_time) / min(eff_time, mob_time)

    print(f'   EfficientNetB0 : {eff_time:.3f}s  ({eff_time/n_test*1000:.1f} ms/image)')
    print(f'   MobileNetV2    : {mob_time:.3f}s  ({mob_time/n_test*1000:.1f} ms/image)')
    print(f'   🏆 {faster} is {ratio:.1f}x faster')

    # Final comparison chart
    plot_final_comparison(val_results)

    # Summary table
    print('\n' + '=' * 70)
    print('📊  FULL RESULTS SUMMARY')
    print('=' * 70)
    print(f"  {'Experiment':<38} {'EfficientNet':>12} {'MobileNet':>12}  {'Set'}")
    print('-' * 70)
    print(f"  {'Phase 1 (frozen base)':<38} "
          f"{val_results['eff_p1']*100:>11.2f}% {val_results['mob_p1']*100:>11.2f}%   val")
    print(f"  {'Strategy A (last N layers)':<38} "
          f"{val_results['eff_a']*100:>11.2f}% {val_results['mob_a']*100:>11.2f}%   val")
    print(f"  {'Strategy B (full blocks)':<38} "
          f"{val_results['eff_b']*100:>11.2f}% {val_results['mob_b']*100:>11.2f}%   val")
    print('-' * 70)
    print(f"  {'Best strategy — FINAL TEST ACC':<38} "
          f"{test_results['eff']*100:>11.2f}% {test_results['mob']*100:>11.2f}%   ✅ test")
    print(f"  {'  (strategy chosen)':<38} "
          f"{'Strat ' + test_results['eff_strategy']:>12} "
          f"{'Strat ' + test_results['mob_strategy']:>12}")
    print('=' * 70)

    best_name = 'EfficientNetB0' if test_results['eff'] >= test_results['mob'] else 'MobileNetV2'
    best_acc  = max(test_results['eff'], test_results['mob'])
    best_strat = test_results['eff_strategy'] if test_results['eff'] >= test_results['mob'] else test_results['mob_strategy']
    print(f'  🏆  Best final test accuracy : {best_name}  '
          f'({best_acc*100:.2f}%)  [strategy {best_strat}]')

    # Architecture comparison
    print('\n' + '=' * 65)
    print('🏗️   ARCHITECTURE COMPARISON')
    print('=' * 65)

    eff_tmp = efficientnet_b0(weights=None)
    mob_tmp = mobilenet_v2(weights=None)

    eff_params = sum(p.numel() for p in eff_tmp.parameters())
    mob_params = sum(p.numel() for p in mob_tmp.parameters())

    rows = [
        ('Base params',        f'{eff_params:,}',      f'{mob_params:,}'),
        ('Architecture',       'MBConv blocks',         'Inv. Residuals'),
        ('Input normalisation','ImageNet μ/σ',          'ImageNet μ/σ'),
        ('Feature blocks',     '9  (features[0-8])',    '19 (features[0-18])'),
        ('Inference speed',    'Moderate',              'Fast ⚡'),
        ('Accuracy potential', 'Higher ✅',             'Slightly lower'),
        ('Memory usage',       'Moderate',              'Lower ✅'),
        ('Speed (measured)',   f'{eff_time:.3f}s',      f'{mob_time:.3f}s'),
    ]
    print(f"  {'Property':<28} {'EfficientNetB0':>16} {'MobileNetV2':>14}")
    print('  ' + '-' * 60)
    for prop, ev, mv in rows:
        print(f"  {prop:<28} {ev:>16} {mv:>14}")
    print('=' * 65)

    del eff_tmp, mob_tmp

# =============================================================
#  GRAD-CAM VISUALIZATION
# =============================================================

import cv2

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Hook for gradients
        self.target_layer.register_full_backward_hook(self.save_gradient)

        # Hook for activations
        self.target_layer.register_forward_hook(self.save_activation)

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def save_activation(self, module, input, output):
        self.activations = output

    def generate(self, input_tensor, class_idx=None):

        self.model.eval()

        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()

        loss = output[:, class_idx]
        loss.backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        pooled_gradients = torch.mean(
            gradients,
            dim=[1, 2]
        )

        for i in range(len(pooled_gradients)):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).cpu().detach().numpy()

        heatmap = np.maximum(heatmap, 0)

        if np.max(heatmap) != 0:
          heatmap /= np.max(heatmap)

        return heatmap, class_idx


def show_gradcam(model, image_path, transform,
                 target_layer, class_names, save_name='gradcam.png'):

    # Load image
    img = Image.open(image_path).convert('RGB')

    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # Create GradCAM object
    cam = GradCAM(model, target_layer)

    # Generate heatmap
    heatmap, pred_class = cam.generate(input_tensor)

    # Convert PIL image → OpenCV format
    img_cv = np.array(img)
    img_cv = cv2.resize(img_cv, (IMG_SIZE, IMG_SIZE))

    # Resize heatmap
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay
    superimposed = heatmap * 0.4 + img_cv

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_cv)
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(heatmap)
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')

    axes[2].imshow(superimposed.astype(np.uint8))
    axes[2].set_title(
        f'Focus Area\nPrediction: {class_names[pred_class]}'
    )
    axes[2].axis('off')

    plt.tight_layout()

    plt.savefig(save_name, dpi=150)

    plt.close()

    print(f'✅ Saved: {save_name}')

# =============================================================
#  MAIN
# =============================================================

def main():

    # Prepare dataset
    prepare_dataset(DATASET_DIR)

    # EDA
    print('\n' + '=' * 60)
    print('  EDA')
    print('=' * 60)
    run_eda(DATA_DIR)

    # Data loaders (single set — both models use ImageNet normalisation)
    print('Creating data loaders...')
    train_loader, val_loader, test_loader, class_to_idx = make_loaders()
    print(f'Class indices: {class_to_idx}\n')

    # Print block boundaries
    print_block_boundaries_eff()
    print_block_boundaries_mob()

    # ─────────────────────────────────────────────────────
    #  PART 1 — Frozen base
    # ─────────────────────────────────────────────────────
    (eff_model, eff_hist1, eff_p1_acc,
     mob_model, mob_hist1, mob_p1_acc) = part1(
        train_loader, val_loader          # ← no test_loader
    )

    # ─────────────────────────────────────────────────────
    #  PART 2 — Unfreezing strategies
    #  All accuracies here are on val_loader.
    #  test_loader is NOT used until after the best model is chosen.
    # ─────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('  PART 2 — Unfreezing Strategies  (val set)')
    print('=' * 60)

    eff_model_a, eff_hist_a, eff_a_val = strategy_a_eff(
        train_loader, val_loader, n_layers=10)

    mob_model_a, mob_hist_a, mob_a_val = strategy_a_mob(
        train_loader, val_loader, n_layers=10)

    eff_model_b, eff_hist_b, eff_b_val = strategy_b_eff(
        train_loader, val_loader, unfreeze_from_block=6)

    mob_model_b, mob_hist_b, mob_b_val = strategy_b_mob(
        train_loader, val_loader, unfreeze_from_block=15)

    # Strategy comparison plots
    plot_strategies(eff_hist_a, eff_hist_b, 'EfficientNetB0')
    plot_strategies(mob_hist_a, mob_hist_b, 'MobileNetV2')
    # TRAINING CURVES
    plot_training_curves(eff_hist_a, 'EfficientNet Strategy A')
    plot_training_curves(eff_hist_b, 'EfficientNet Strategy B')
    plot_training_curves(mob_hist_a, 'MobileNet Strategy A')
    plot_training_curves(mob_hist_b, 'MobileNet Strategy B')
    # Learning rate plots
    plot_learning_rate(eff_hist_a, 'EfficientNet Strategy A')
    plot_learning_rate(eff_hist_b, 'EfficientNet Strategy B')

    plot_learning_rate(mob_hist_a, 'MobileNet Strategy A')
    plot_learning_rate(mob_hist_b, 'MobileNet Strategy B')

    # ── Pick best per model using val accuracy ────────────
    best_eff = eff_model_b if eff_b_val >= eff_a_val else eff_model_a
    best_mob = mob_model_b if mob_b_val >= mob_a_val else mob_model_a
    best_eff_strategy = 'B' if eff_b_val >= eff_a_val else 'A'
    best_mob_strategy = 'B' if mob_b_val >= mob_a_val else 'A'
    print(f'\n🏆  Best EfficientNet strategy (by val): '
          f'{best_eff_strategy} ({max(eff_a_val, eff_b_val)*100:.2f}%)')
    print(f'🏆  Best MobileNet strategy   (by val): '
          f'{best_mob_strategy} ({max(mob_a_val, mob_b_val)*100:.2f}%)')

    # ── Single final evaluation on test set ───────────────
    # This is the only place test_loader is used. It runs once,
    # after all decisions have been made using val_loader only.
    print('\n' + '─' * 60)
    print('  🔒  Final Test Evaluation  (test set, used once)')
    print('─' * 60)
    best_eff_test = test_accuracy(best_eff, test_loader)
    best_mob_test = test_accuracy(best_mob, test_loader)
    print(f'  EfficientNetB0 (Strategy {best_eff_strategy}) test accuracy: '
          f'{best_eff_test*100:.2f}%')
    print(f'  MobileNetV2    (Strategy {best_mob_strategy}) test accuracy: '
          f'{best_mob_test*100:.2f}%')

    # ─────────────────────────────────────────────────────
    #  PART 3 — Analysis
    # ─────────────────────────────────────────────────────
    # val_accs  → used for strategy comparison table (Part 2)
    # test_accs → the single honest final number per model
    val_results = dict(
        eff_p1=eff_p1_acc, eff_a=eff_a_val, eff_b=eff_b_val,
        mob_p1=mob_p1_acc, mob_a=mob_a_val, mob_b=mob_b_val,
    )
    test_results = dict(
        eff=best_eff_test,
        mob=best_mob_test,
        eff_strategy=best_eff_strategy,
        mob_strategy=best_mob_strategy,
    )
    part3_analysis(val_results, test_results, best_eff, best_mob,
                   test_loader, class_to_idx)

    # Save best models
    torch.save(best_eff.state_dict(), 'thobe_efficientnet_best.pt')
    torch.save(best_mob.state_dict(), 'thobe_mobilenet_best.pt')
    print('\n✅  Saved: thobe_efficientnet_best.pt')
    print('✅  Saved: thobe_mobilenet_best.pt')

    print('\n' + '=' * 60)
    print('  ✅  All done! Check saved .png and .pt files.')
    print('=' * 60)

    print('\n' + '=' * 60)
    print('  ✅  All done! Check saved .png and .pt files.')
    # =====================================================
    # GRAD-CAM VISUALIZATION
    # =====================================================

    _, val_tf = make_transforms()

    show_gradcam(
        model=best_eff,
        image_path='Dataset/jaffa/ChatGPT Image May 8, 2026, 06_37_32 PM (1).png',
        transform=val_tf,
        target_layer=best_eff.features[-1],
        class_names=CLASSES,
        save_name='gradcam_eff.png'
    )

    show_gradcam(
        model=best_mob,
        image_path='Dataset/jaffa/ChatGPT Image May 8, 2026, 06_37_32 PM (1).png',
        transform=val_tf,
        target_layer=best_mob.features[-1],
        class_names=CLASSES,
        save_name='gradcam_mob.png'
    )

# =============================================================

if __name__ == '__main__':
    main()