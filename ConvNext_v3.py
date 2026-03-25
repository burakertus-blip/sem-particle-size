
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

BASE_DIR = r"."

PART_DIR = os.path.join(BASE_DIR, "synthetic_sem_dataset")
PART_CSV = os.path.join(PART_DIR, "labels.csv")

PART_MODEL_PATH = os.path.join(BASE_DIR, "convnext_particle_triplet.pth")
PART_BEST_PATH  = os.path.join(BASE_DIR, "convnext_particle_triplet_best.pth")
PART_EMB_PATH   = os.path.join(BASE_DIR, "emb_particle_triplet.npz")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'CPU'}")

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)

LR = 5e-5
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 25

BATCH_SIZE = 8 if DEVICE == "cuda" else 4
GRAD_ACCUM = 2

CACHE_IMAGES = True
if CACHE_IMAGES:
    NUM_WORKERS = 0
    PERSISTENT_WORKERS = False
    PREFETCH_FACTOR = None
    print("RAM cache enabled: NUM_WORKERS=0.")
else:
    NUM_WORKERS = 4 if DEVICE == "cuda" else 0
    PERSISTENT_WORKERS = True if NUM_WORKERS > 0 else False
    PREFETCH_FACTOR = 2 if NUM_WORKERS > 0 else None

USE_CHANNELS_LAST = True

TRIPLET_MARGIN = 0.25
USE_REGRESSION_HEAD = True
REG_WEIGHT = 0.25

FREEZE_EPOCHS = 5
UNFREEZE_LAST_N_BLOCKS = 2

LOG_EPS = 1e-3
HUBER_BETA = 0.05

VAL_FRAC = 0.15

IMG_COL_CANDS = ["image_name", "imgname", "filename", "file"]
PART_TARGET_CANDS = ["particle_size_ratio", "particle size", "particle_size", "size"]

def read_csv_safe(path: str) -> pd.DataFrame:
    """
    Robust CSV reader:
    - tries sep=None (auto-detect) with python engine
    - then tries ';'
    - then tries fallback split if single header contains ';'
    """
    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, sep=None, engine="python")
        if df is not None and not df.empty and df.shape[1] > 1:
            return df
    except Exception:
        pass

    try:
        df = pd.read_csv(path, sep=";")
        if df is not None and not df.empty and df.shape[1] > 1:
            return df
    except Exception:
        pass

    try:
        df = pd.read_csv(path)
        if df is None or df.empty:
            return pd.DataFrame()
        if df.shape[1] == 1:
            col = df.columns[0]
            if ";" in col:
                new_cols = [c.strip() for c in col.split(";")]
                df2 = df[col].astype(str).str.split(";", expand=True)
                if df2.shape[1] == len(new_cols):
                    df2.columns = new_cols
                    return df2
    except Exception:
        pass

    return pd.DataFrame()

def find_col(df, candidates):
    if df is None or df.empty:
        return None
    lower_map = {str(c).lower().strip(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().strip()
        if key in lower_map:
            return lower_map[key]
    return None

def ensure_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def list_images_in_dir(img_dir):
    d = {}
    if not os.path.isdir(img_dir):
        return d
    for fn in os.listdir(img_dir):
        if fn.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")):
            d[fn] = os.path.join(img_dir, fn)
    return d

def load_gray_to_3ch(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return np.stack([img, img, img], axis=-1)

def get_transforms_train():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
    ])

def get_transforms_eval():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]),
    ])

def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def build_loader(ds, shuffle=True):
    kwargs = dict(
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        collate_fn=safe_collate,
        drop_last=shuffle,
    )
    if NUM_WORKERS > 0:
        kwargs["persistent_workers"] = PERSISTENT_WORKERS
        kwargs["prefetch_factor"] = PREFETCH_FACTOR
    return DataLoader(ds, **kwargs)

def split_df(df, val_frac=0.15, seed=42):
    idx = np.arange(len(df))
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    n_val = max(1, int(round(val_frac * len(df))))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return df.iloc[tr_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)

class SingleImageSEM(Dataset):
    def __init__(self, df, img_dir, img_col, target_col, transform=None, cache_images=False,
                 y_mu=None, y_std=None, use_log=True):
        self.df = df.reset_index(drop=True).copy()
        self.img_dir = img_dir
        self.img_col = img_col
        self.target_col = target_col
        self.transform = transform
        self.img_map = list_images_in_dir(img_dir)

        keep = []
        self.fns = []
        self.y_raw = []
        for i in range(len(self.df)):
            fn = str(self.df.loc[i, self.img_col]).strip()
            y = ensure_float(self.df.loc[i, self.target_col])
            if fn in self.img_map and not np.isnan(y):
                keep.append(i)
                self.fns.append(fn)
                self.y_raw.append(float(y))

        self.df = self.df.loc[keep].reset_index(drop=True)
        self.y_raw = np.array(self.y_raw, dtype=np.float32)

        self.use_log = use_log
        if self.use_log:
            y_t = np.log(self.y_raw + LOG_EPS)
        else:
            y_t = self.y_raw.copy()

        if y_mu is None:
            self.y_mu = float(np.mean(y_t))
        else:
            self.y_mu = float(y_mu)
        if y_std is None:
            self.y_std = float(np.std(y_t) + 1e-8)
        else:
            self.y_std = float(y_std)

        self.y = ((y_t - self.y_mu) / self.y_std).astype(np.float32)

        self.cache_images = cache_images
        self.cache = {}
        if self.cache_images:
            print(f"Loading into RAM cache: {os.path.basename(img_dir)} ({len(self.df)} images) ...")
            for fn in sorted(set(self.fns)):
                path = self.img_map.get(fn)
                if path:
                    img = load_gray_to_3ch(path)
                    if img is not None:
                        self.cache[fn] = img
            print(f"Cache ready: {len(self.cache)} images.")

    def __len__(self):
        return len(self.df)

    def _get_img(self, fn):
        if self.cache_images and fn in self.cache:
            return self.cache[fn]
        path = self.img_map.get(fn)
        if not path:
            return None
        return load_gray_to_3ch(path)

    def __getitem__(self, idx):
        fn = self.fns[idx]
        y = float(self.y[idx])
        y_raw = float(self.y_raw[idx])
        img = self._get_img(fn)
        if img is None:
            return None
        x = self.transform(img) if self.transform else torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return x, torch.tensor([y], dtype=torch.float32), torch.tensor([y_raw], dtype=torch.float32), fn

class ConvNeXtEmbedder(nn.Module):
    def __init__(self, embedding_dim=256, use_regression=True):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        self.backbone = models.convnext_tiny(weights=weights)
        self.backbone.classifier = nn.Identity()

        self.fc_embed = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, embedding_dim),
            nn.GELU()
        )

        self.use_regression = use_regression
        if self.use_regression:
            self.fc_out = nn.Sequential(
                nn.Linear(embedding_dim, 128),
                nn.GELU(),
                nn.Linear(128, 1)
            )

    def forward(self, x):
        x = self.backbone(x).flatten(1)
        emb = self.fc_embed(x)
        if self.use_regression:
            pred = self.fc_out(emb)
            return pred, emb
        return None, emb

def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

def freeze_backbone(model: ConvNeXtEmbedder):
    set_requires_grad(model.backbone, False)
    set_requires_grad(model.fc_embed, True)
    if model.use_regression:
        set_requires_grad(model.fc_out, True)

def unfreeze_last_blocks(model: ConvNeXtEmbedder, n_blocks=2):
    set_requires_grad(model.backbone, False)
    if hasattr(model.backbone, "features"):
        feats = model.backbone.features
        total = len(feats)
        start = max(0, total - n_blocks)
        for i in range(start, total):
            set_requires_grad(feats[i], True)
    set_requires_grad(model.fc_embed, True)
    if model.use_regression:
        set_requires_grad(model.fc_out, True)

def batch_hard_triplet_loss(emb, y, margin=0.25):
    """
    AMP-safe:
    - forces ALL mining tensors to fp32 before masked_fill
    - prevents fp16 overflow completely
    """
    B = emb.size(0)
    if B < 3:
        z = torch.tensor(0.0, device=emb.device)
        return z, (z, z)

    device = emb.device

    emb32 = emb.detach().float()
    yv32  = y.view(-1).detach().float()

    sim32 = emb32 @ emb32.t()
    dist_emb32 = (1.0 - sim32).clamp(min=0.0, max=2.0)

    dist_y32 = torch.abs(yv32[:, None] - yv32[None, :])

    eye = torch.eye(B, device=device, dtype=torch.bool)
    BIG = 1e6
    dist_y32 = dist_y32.masked_fill(eye, BIG)

    pos_idx = torch.argmin(dist_y32, dim=1)
    d_ap = dist_emb32[torch.arange(B, device=device), pos_idx]

    valid = dist_y32[dist_y32 < BIG]
    if valid.numel() == 0:
        neg_idx = torch.argmax(dist_y32, dim=1)
    else:
        thresh = torch.median(valid)
        neg_mask = dist_y32 >= thresh

        if neg_mask.sum() < B:
            neg_idx = torch.argmax(dist_y32, dim=1)
        else:
            masked = dist_emb32.masked_fill(~neg_mask, BIG)
            neg_idx = torch.argmin(masked, dim=1)

    d_an = dist_emb32[torch.arange(B, device=device), neg_idx]
    loss = F.relu(d_ap - d_an + float(margin)).mean()

    return loss, (d_ap.mean().detach(), d_an.mean().detach())

def inv_transform(y_hat_std, y_mu, y_std, use_log=True):
    y_t = y_hat_std * y_std + y_mu
    y_t = y_t.detach().cpu().numpy().astype(np.float64)
    if use_log:
        y_raw = np.exp(y_t) - LOG_EPS
    else:
        y_raw = y_t
    return y_raw

def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12)
    r2 = 1.0 - ss_res / ss_tot
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.abs((y_true - y_pred) / y_true) * 100.0
        m = m[np.isfinite(m)]
        mape = float(np.mean(m)) if len(m) else 0.0
    return mae, mape, rmse, r2

@torch.no_grad()
def eval_regression(model, loader, y_mu, y_std, use_log=True):
    model.eval()
    y_true_raw_all, y_pred_raw_all = [], []
    for batch in loader:
        if batch is None:
            continue
        x, _, y_raw, _ = batch
        x = x.to(DEVICE, non_blocking=True)
        if USE_CHANNELS_LAST:
            x = x.to(memory_format=torch.channels_last)

        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            p, _ = model(x)

        y_pred_raw = inv_transform(p.view(-1), y_mu, y_std, use_log=use_log)
        y_true_raw = y_raw.view(-1).cpu().numpy().astype(np.float64)
        y_true_raw_all.append(y_true_raw)
        y_pred_raw_all.append(y_pred_raw)

    if not y_true_raw_all:
        return None
    y_true_raw_all = np.concatenate(y_true_raw_all)
    y_pred_raw_all = np.concatenate(y_pred_raw_all)
    return compute_metrics(y_true_raw_all, y_pred_raw_all)

def train_particle(df_part, img_col, tgt_col):
    tf_train = get_transforms_train()
    tf_eval  = get_transforms_eval()

    df_tr, df_va = split_df(df_part, val_frac=VAL_FRAC, seed=SEED)
    print(f"Split: train={len(df_tr)} | val={len(df_va)}")

    ds_tr_tmp = SingleImageSEM(df_tr, PART_DIR, img_col, tgt_col, transform=tf_train,
                               cache_images=CACHE_IMAGES, y_mu=None, y_std=None, use_log=True)
    y_mu, y_std = ds_tr_tmp.y_mu, ds_tr_tmp.y_std
    print(f"Target: log(y+{LOG_EPS}) then z-score | mu={y_mu:.4f}, std={y_std:.4f}")

    ds_tr = SingleImageSEM(df_tr, PART_DIR, img_col, tgt_col, transform=tf_train,
                           cache_images=CACHE_IMAGES, y_mu=y_mu, y_std=y_std, use_log=True)
    ds_va = SingleImageSEM(df_va, PART_DIR, img_col, tgt_col, transform=tf_eval,
                           cache_images=CACHE_IMAGES, y_mu=y_mu, y_std=y_std, use_log=True)

    dl_tr = build_loader(ds_tr, shuffle=True)
    dl_va = build_loader(ds_va, shuffle=False)

    model = ConvNeXtEmbedder(embedding_dim=256, use_regression=USE_REGRESSION_HEAD).to(DEVICE)
    if USE_CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)
        print("Channels Last enabled")

    freeze_backbone(model)
    print(f"Backbone frozen: first {FREEZE_EPOCHS} epochs")

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=NUM_EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    best_val_mae = float("inf")
    triplet_margin = TRIPLET_MARGIN

    for epoch in range(1, NUM_EPOCHS + 1):
        if epoch == FREEZE_EPOCHS + 1:
            unfreeze_last_blocks(model, n_blocks=UNFREEZE_LAST_N_BLOCKS)
            print(f"Unfreezing: last {UNFREEZE_LAST_N_BLOCKS} backbone blocks at epoch {epoch}")

            opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=LR * 0.7, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=(NUM_EPOCHS - epoch + 1))

        model.train()
        opt.zero_grad(set_to_none=True)

        tot, t_tri, t_reg = 0.0, 0.0, 0.0
        dap_m, dan_m = 0.0, 0.0
        steps = 0

        for i, batch in enumerate(dl_tr):
            if batch is None:
                continue
            x, y_std_t, _, _ = batch
            x = x.to(DEVICE, non_blocking=True)
            y_std_t = y_std_t.to(DEVICE, non_blocking=True)

            if USE_CHANNELS_LAST:
                x = x.to(memory_format=torch.channels_last)

            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                p, emb = model(x)
                emb_n = F.normalize(emb, p=2, dim=1)

            with torch.cuda.amp.autocast(enabled=False):
                loss_tri, (dap, dan) = batch_hard_triplet_loss(emb_n, y_std_t, margin=triplet_margin)

            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                if model.use_regression:
                    loss_reg = F.smooth_l1_loss(p, y_std_t, beta=HUBER_BETA)
                    loss = loss_tri + REG_WEIGHT * loss_reg
                else:
                    loss_reg = torch.tensor(0.0, device=DEVICE)
                    loss = loss_tri
                loss = loss / GRAD_ACCUM

            scaler.scale(loss).backward()

            if (i + 1) % GRAD_ACCUM == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            tot += float(loss.item()) * GRAD_ACCUM
            t_tri += float(loss_tri.item())
            t_reg += float(loss_reg.item()) if model.use_regression else 0.0
            dap_m += float(dap.item())
            dan_m += float(dan.item())
            steps += 1

        scheduler.step()

        if steps == 0:
            print(f"Ep {epoch:02d} | no steps")
            continue

        val_metrics = eval_regression(model, dl_va, y_mu, y_std, use_log=True)
        if val_metrics is None:
            print(f"Ep {epoch:02d} | TrainLoss {tot/steps:.4f} | (val metrics unavailable)")
            continue

        val_mae, val_mape, val_rmse, val_r2 = val_metrics

        print(
            f"Ep {epoch:02d} | "
            f"TrainLoss {tot/steps:.4f} | Tri {t_tri/steps:.4f} | Reg {t_reg/steps:.4f} | "
            f"d_ap {dap_m/steps:.4f} | d_an {dan_m/steps:.4f} | "
            f"VAL: MAE {val_mae:.4f} | MAPE {val_mape:.2f}% | RMSE {val_rmse:.4f} | R2 {val_r2:.4f}"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                "state_dict": model.state_dict(),
                "y_mu": y_mu,
                "y_std": y_std,
                "use_log": True,
                "log_eps": LOG_EPS
            }, PART_BEST_PATH)
            print(f"BEST checkpoint saved (VAL MAE): {PART_BEST_PATH} | MAE={best_val_mae:.4f}")

    torch.save({
        "state_dict": model.state_dict(),
        "y_mu": y_mu,
        "y_std": y_std,
        "use_log": True,
        "log_eps": LOG_EPS
    }, PART_MODEL_PATH)
    print(f"Final model saved: {PART_MODEL_PATH}")

    return model, y_mu, y_std

@torch.no_grad()
def save_embeddings(model, df_all, img_col, tgt_col, path, y_mu, y_std):
    print(f"Extracting embeddings: {path}")
    tf_eval = get_transforms_eval()

    ds = SingleImageSEM(df_all, PART_DIR, img_col, tgt_col, transform=tf_eval,
                        cache_images=CACHE_IMAGES, y_mu=y_mu, y_std=y_std, use_log=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=safe_collate)

    model.eval()

    embs, tgts, names = [], [], []
    for batch in loader:
        if batch is None:
            continue
        x, _, y_raw, fn = batch
        x = x.to(DEVICE)
        if USE_CHANNELS_LAST:
            x = x.to(memory_format=torch.channels_last)

        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            _, emb = model(x)

        emb = F.normalize(emb, p=2, dim=1)
        embs.append(emb.detach().cpu().numpy())
        tgts.append(y_raw.cpu().numpy())
        names.append(fn[0])

    np.savez(
        path,
        embeddings=np.concatenate(embs, axis=0),
        targets=np.concatenate(tgts, axis=0),
        imgnames=np.array(names)
    )
    print("Embeddings saved.")

def main():
    df = read_csv_safe(PART_CSV)
    if df.empty:
        print("Particle CSV not found or empty:", PART_CSV)
        return

    df.columns = [str(c).strip() for c in df.columns]

    img_col = find_col(df, IMG_COL_CANDS)
    tgt_col = find_col(df, PART_TARGET_CANDS)

    if img_col is None or tgt_col is None:
        print("Image/target column not found in CSV.")
        print("target column:", list(df.columns))
        print("If CSV delimiter is semicolon, check labels.csv delimiter.")
        return

    print(f"image_col: {img_col} | target_col: {tgt_col}")

    model, y_mu, y_std = train_particle(df, img_col, tgt_col)
    save_embeddings(model, df, img_col, tgt_col, PART_EMB_PATH, y_mu, y_std)

if __name__ == "__main__":
    main()
