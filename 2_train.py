"""
2_train.py
Trains the 2D U-Net on all 12 labeled subjects (no validation split).
All prepared NPZ files in data/prepared/ are pooled into one dataset.
Positive slices (has_defect=True) are oversampled 5x to handle class imbalance.

Outputs:
  models/best_model.pth   — checkpoint with lowest training Dice loss
  models/last_model.pth   — checkpoint saved every 10 epochs
  logs/train_log.csv      — per-epoch loss and Dice score
"""

import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from model import UNet, combined_loss, count_parameters

SCRIPT_DIR   = Path(__file__).parent
PREPARED_DIR = SCRIPT_DIR / 'data' / 'prepared'
MODELS_DIR   = SCRIPT_DIR / 'models'
LOGS_DIR     = SCRIPT_DIR / 'logs'
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

EPOCHS        = 100
BATCH_SIZE    = 8
LR            = 1e-4
TRAIN_SIZE    = 256
POS_OVERSAMPLE = 5
SAVE_EVERY     = 10
RANDOM_SEED    = 42


def compute_otsu_map(img_norm: np.ndarray) -> np.ndarray:
    """
    Binary bone mask via Otsu's method on a [0,1]-normalised image.
    Used as the second input channel so the model can leverage bone/non-bone
    boundary information learned from the Otsu threshold.
    Pure NumPy — no additional dependencies.
    Returns float32 array with values 0.0 (background) or 1.0 (bone/foreground).
    """
    hist, edges = np.histogram(img_norm.ravel(), bins=256, range=(0.0, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = float(hist.sum())
    if total == 0:
        return np.zeros_like(img_norm, dtype=np.float32)
    p    = hist.astype(np.float64) / total
    w0   = np.cumsum(p)
    w1   = 1.0 - w0
    mc   = np.cumsum(p * centers)
    mT   = mc[-1]
    mu0  = mc / np.maximum(w0, 1e-10)
    mu1  = (mT - mc) / np.maximum(w1, 1e-10)
    sigma = w0 * w1 * (mu0 - mu1) ** 2
    thresh = float(centers[np.argmax(sigma)])
    return (img_norm >= thresh).astype(np.float32)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class DefectDataset(Dataset):
    """Loads all slices from a list of (image, mask, otsu_map) numpy array triples."""

    def __init__(self, images: np.ndarray, masks: np.ndarray,
                 otsu_maps: np.ndarray, augment: bool = True):
        self.images    = images     # (N, 512, 512) float32
        self.masks     = masks      # (N, 512, 512) uint8
        self.otsu_maps = otsu_maps  # (N, 512, 512) float32 binary
        self.augment   = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img  = self.images[idx].copy()     # (512, 512) float32
        mask = self.masks[idx].copy()      # (512, 512) uint8
        otsu = self.otsu_maps[idx].copy()  # (512, 512) float32

        # Downsample 512 → TRAIN_SIZE for faster training (nearest-neighbor)
        s = 512 // TRAIN_SIZE
        if s > 1:
            img  = img[::s, ::s]
            otsu = otsu[::s, ::s]
            # For mask use max-pool to avoid missing thin edges
            h, w = mask.shape
            mask = mask.reshape(h // s, s, w // s, s).max(axis=(1, 3))

        if self.augment:
            img, mask, otsu = self._augment(img, mask, otsu)

        # Channel 0: HU-normalised image; channel 1: Otsu bone binary map
        inp_t  = torch.from_numpy(np.stack([img, otsu], axis=0))  # (2, H, W)
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)  # (1, H, W)
        return inp_t, mask_t

    @staticmethod
    def _augment(img, mask, otsu):
        # Horizontal flip
        if random.random() < 0.5:
            img  = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
            otsu = np.fliplr(otsu).copy()
        # Vertical flip
        if random.random() < 0.5:
            img  = np.flipud(img).copy()
            mask = np.flipud(mask).copy()
            otsu = np.flipud(otsu).copy()
        # 90-degree rotation (0, 90, 180, 270)
        k = random.randint(0, 3)
        if k > 0:
            img  = np.rot90(img,  k).copy()
            mask = np.rot90(mask, k).copy()
            otsu = np.rot90(otsu, k).copy()
        # Intensity jitter on HU channel only (Otsu map stays binary)
        if random.random() < 0.5:
            delta = random.uniform(-0.08, 0.08)
            img   = np.clip(img + delta, 0.0, 1.0)
        if random.random() < 0.5:
            scale = random.uniform(0.9, 1.1)
            img   = np.clip(img * scale, 0.0, 1.0)
        return img, mask, otsu


def load_all_npz(prepared_dir: Path):
    """Pool all subject NPZ files into flat arrays including otsu_maps."""
    all_images     = []
    all_masks      = []
    all_otsu_maps  = []
    all_has_defect = []
    subject_ids    = []

    npz_files = sorted(f for f in prepared_dir.glob('*.npz') if not f.name.startswith('._'))
    if not npz_files:
        raise FileNotFoundError(f'No .npz files in {prepared_dir}. Run 1_prepare_dataset.py first.')

    print(f'Loading {len(npz_files)} subject NPZ files...')
    for npz_path in npz_files:
        data = np.load(str(npz_path))
        n = len(data['images'])
        all_images.append(data['images'])
        all_masks.append(data['masks'])
        all_has_defect.append(data['has_defect'])
        if 'otsu_maps' in data:
            all_otsu_maps.append(data['otsu_maps'])
        else:
            print(f'  {npz_path.name}: otsu_maps missing — computing on-the-fly (re-run 1_prepare_dataset.py to cache)')
            all_otsu_maps.append(
                np.stack([compute_otsu_map(img) for img in data['images']], axis=0)
            )
        sid = npz_path.stem
        subject_ids.extend([sid] * n)
        print(f'  {npz_path.name}: {n} slices  (pos={data["has_defect"].sum()})')

    images     = np.concatenate(all_images,     axis=0)
    masks      = np.concatenate(all_masks,      axis=0)
    otsu_maps  = np.concatenate(all_otsu_maps,  axis=0)
    has_defect = np.concatenate(all_has_defect, axis=0)

    print(f'\nTotal slices : {len(images)}')
    print(f'Positive     : {has_defect.sum()} ({100*has_defect.mean():.1f}%)')
    print(f'Negative     : {(~has_defect).sum()}')
    return images, masks, has_defect, otsu_maps


def build_sampler(has_defect: np.ndarray, pos_weight: int = POS_OVERSAMPLE):
    """WeightedRandomSampler oversampling positive slices."""
    weights = np.where(has_defect, float(pos_weight), 1.0)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=len(weights),
        replacement=True,
    )
    return sampler


def dice_score_batch(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    with torch.no_grad():
        probs = (torch.sigmoid(logits) > 0.5).float()
        probs   = probs.view(-1)
        targets = targets.view(-1).float()
        inter = (probs * targets).sum().item()
        union = probs.sum().item() + targets.sum().item()
        return (2 * inter + eps) / (union + eps)


def train_one_epoch(model, loader, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    n_batches  = 0
    n_total    = len(loader)

    for imgs, masks in loader:
        imgs  = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = combined_loss(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice += dice_score_batch(logits, masks)
        n_batches  += 1
        if n_batches % 50 == 0:
            print(f'  Ep {epoch}/{total_epochs}  batch {n_batches}/{n_total}')

    return total_loss / n_batches, total_dice / n_batches


def main():
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = get_device()
    print(f'Device: {device}')

    images, masks, has_defect, otsu_maps = load_all_npz(PREPARED_DIR)

    dataset = DefectDataset(images, masks, otsu_maps, augment=True)
    sampler = build_sampler(has_defect)
    loader  = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
        drop_last=True,
    )

    model = UNet(in_channels=2, out_channels=1, features=(32, 64, 128, 256))
    model = model.to(device)
    print(f'Model parameters: {count_parameters(model):,}')

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    log_path = LOGS_DIR / 'train_log.csv'
    log_file = open(log_path, 'w', newline='')
    log_writer = csv.DictWriter(log_file, fieldnames=['epoch', 'loss', 'dice', 'lr', 'elapsed_s'])
    log_writer.writeheader()

    best_loss   = float('inf')
    start_time  = time.time()

    print(f'\nTraining for {EPOCHS} epochs  |  batch={BATCH_SIZE}  |  lr={LR}')
    print('─' * 60)

    for epoch in range(1, EPOCHS + 1):
        ep_loss, ep_dice = train_one_epoch(model, loader, optimizer, device, epoch, EPOCHS)
        scheduler.step()
        elapsed = time.time() - start_time
        lr_now  = scheduler.get_last_lr()[0]

        log_writer.writerow({
            'epoch': epoch, 'loss': f'{ep_loss:.6f}',
            'dice': f'{ep_dice:.4f}', 'lr': f'{lr_now:.2e}',
            'elapsed_s': f'{elapsed:.1f}',
        })
        log_file.flush()

        print(f'Epoch {epoch:3d}/{EPOCHS}  loss={ep_loss:.4f}  dice={ep_dice:.4f}'
              f'  lr={lr_now:.2e}  [{elapsed/60:.1f} min]')

        if ep_loss < best_loss:
            best_loss = ep_loss
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'loss': ep_loss, 'dice': ep_dice},
                       MODELS_DIR / 'best_model.pth')
            print(f'  ↑ best model saved (loss={best_loss:.4f})')

        if epoch % SAVE_EVERY == 0:
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'loss': ep_loss, 'dice': ep_dice},
                       MODELS_DIR / 'last_model.pth')

    log_file.close()
    print(f'\nTraining complete. Best loss: {best_loss:.4f}')
    print(f'Model     → {MODELS_DIR / "best_model.pth"}')
    print(f'Train log → {log_path}')


if __name__ == '__main__':
    main()
