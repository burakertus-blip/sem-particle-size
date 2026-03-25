import os
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import random
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = r"."
TEST_ROOT  = os.path.join(BASE_DIR, "test")
PART_DIR   = os.path.join(TEST_ROOT, "part")
PART_LABEL = os.path.join(PART_DIR, "part_label.xlsx")

RESULTS_DIR = os.path.join(BASE_DIR, "Paper_Results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_SCALE_PX_CONSTANT = 80.0
TOP_K  = 15
ALPHA  = 0.2
TEMP_T = 0.5
Y_MU   = -0.1571
Y_STD  = 0.9080
LOG_EPS = 1e-3
SCALEBAR_COLOR_BGR = (0, 0, 255)
COLOR_TOLERANCE = 10

PART_MODEL = os.path.join(BASE_DIR, "convnext_particle_triplet.pth")
PART_BEST  = os.path.join(BASE_DIR, "convnext_particle_triplet_best.pth")
PART_EMB   = os.path.join(BASE_DIR, "emb_particle_triplet.npz")

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300
})

class ConvNeXtEmbedder(nn.Module):
    def __init__(self, embedding_dim=256, use_regression=True):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        self.backbone = models.convnext_tiny(weights=weights)
        self.backbone.classifier = nn.Identity()
        self.fc_embed = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, embedding_dim), nn.GELU())
        self.use_regression = use_regression
        if self.use_regression:
            self.fc_out = nn.Sequential(nn.Linear(embedding_dim, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, x):
        x = self.backbone(x).flatten(1)
        emb = self.fc_embed(x)
        if self.use_regression: return self.fc_out(emb), emb
        return None, emb

def load_model_resources():
    path = PART_BEST if os.path.exists(PART_BEST) else PART_MODEL
    if not os.path.exists(path): return None, None, None, None
    print(f"Loading model: {os.path.basename(path)}")
    model = ConvNeXtEmbedder(embedding_dim=256, use_regression=True).to(DEVICE)
    ckpt = torch.load(path, map_location=DEVICE)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    if not os.path.exists(PART_EMB): return None, None, None, None
    d = np.load(PART_EMB, allow_pickle=True)
    tr_embs = F.normalize(torch.from_numpy(d["embeddings"].astype(np.float32)), p=2, dim=1).numpy()
    tr_tgts = d["targets"].astype(np.float64).reshape(-1)
    tr_names = d["imgnames"].astype(str)

    return model, tr_embs, tr_tgts, tr_names

TFM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.25]*3),
])

def process_image(img_path):
    img = cv2.imread(img_path)
    if img is None: return None, 0.0

    lower = np.array([max(c - COLOR_TOLERANCE, 0) for c in SCALEBAR_COLOR_BGR])
    upper = np.array([min(c + COLOR_TOLERANCE, 255) for c in SCALEBAR_COLOR_BGR])
    mask = cv2.inRange(img, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    scale_len = -1.0
    crop_y = img.shape[0]
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        if w > 10:
            scale_len = float(w)
            crop_y = max(0, y - 5)

    img_clean = img[0:crop_y, :]
    g3 = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)
    g3 = np.stack([g3, g3, g3], axis=-1)
    return g3, float(scale_len)

@torch.no_grad()
def predict_image(model, img_arr, scale_px, tr_embs, tr_tgts):
    x = TFM(img_arr).unsqueeze(0).to(DEVICE)
    reg_pred, emb = model(x)

    zlog = reg_pred.item()
    reg_ratio = float(np.exp(float(zlog) * (Y_STD + 1e-12) + Y_MU) - LOG_EPS)

    z_new = F.normalize(emb.squeeze(0), p=2, dim=0).cpu().numpy().astype(np.float32)
    sims = (tr_embs @ z_new.reshape(-1, 1)).reshape(-1)
    idxs = np.arange(len(sims), dtype=np.float64)
    sims_stable = sims.astype(np.float64) - 1e-12 * idxs
    top_idx = np.argsort(-sims_stable)[:TOP_K]

    top_sims = sims[top_idx].astype(np.float64)
    exps = np.exp((top_sims - np.max(top_sims)) / max(float(TEMP_T), 1e-6))
    w = exps / (np.sum(exps) + 1e-12)
    knn_ratio = float(np.sum(tr_tgts[top_idx] * w))

    corr = (TRAIN_SCALE_PX_CONSTANT / scale_px) if scale_px > 10 else 1.0
    final_pred = (ALPHA * reg_ratio + (1.0 - ALPHA) * knn_ratio) * corr

    return final_pred

def plot_scatter_scientific(y_true, y_pred, metrics, save_path):
    plt.figure(figsize=(8, 7))

    plt.scatter(y_pred, y_true, alpha=0.6, edgecolors='w', s=70, color='#4c72b0', label='Test Samples')

    max_val = max(y_true.max(), y_pred.max()) * 1.05
    plt.plot([0, max_val], [0, max_val], color='gray', linestyle='--', linewidth=1.5, label='Ideal (y=x)')

    X = y_pred.reshape(-1, 1)
    reg = LinearRegression().fit(X, y_true)
    y_trend = reg.predict(X)
    plt.plot(y_pred, y_trend, color='#c44e52', linewidth=2, label=f'Best Fit (Slope={reg.coef_[0]:.2f})')

    plt.title(f"Predicted vs. Ground Truth\n(Spearman $\\rho = {metrics['spearman']:.3f}$)", fontweight='bold')
    plt.xlabel("Predicted Size Ratio")
    plt.ylabel("Ground Truth Ratio")
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.legend(loc='upper left')

    textstr = '\n'.join((
        f"N = {len(y_true)}",
        f"Pearson r = {metrics['pearson']:.3f}",
        f"sMAPE = {metrics['smape']:.1f}%",
        f"Bias = {metrics['bias']:.3f}"
    ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.9)
    plt.text(0.95 * max_val, 0.05 * max_val, textstr, fontsize=11,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_bland_altman(y_true, y_pred, save_path):
    means = (y_true + y_pred) / 2
    diffs = y_pred - y_true
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)

    plt.figure(figsize=(8, 6))
    plt.scatter(means, diffs, alpha=0.6, edgecolors='w', s=60, color='purple')

    plt.axhline(mean_diff, color='black', linestyle='-', label=f'Mean Bias ({mean_diff:.3f})')

    upper = mean_diff + 1.96 * std_diff
    lower = mean_diff - 1.96 * std_diff
    plt.axhline(upper, color='red', linestyle='--', label=f'+1.96 SD ({upper:.3f})')
    plt.axhline(lower, color='red', linestyle='--', label=f'-1.96 SD ({lower:.3f})')

    plt.title("Bland-Altman Plot", fontweight='bold')
    plt.xlabel("Mean of Ground Truth and Prediction")
    plt.ylabel("Difference (Predicted - Ground Truth)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_error_hist(y_true, y_pred, save_path):
    pe = ((y_pred - y_true) / y_true) * 100

    plt.figure(figsize=(8, 6))
    sns.histplot(pe, kde=True, bins=15, color='green', edgecolor='black')
    plt.axvline(0, color='black', linestyle='--')
    plt.title("Percentage Error Distribution", fontweight='bold')
    plt.xlabel("Percentage Error (%)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_rec_curve(y_true, y_pred, save_path):
    errors = np.abs(y_pred - y_true)
    tolerances = np.linspace(0, errors.max(), 100)
    accuracy = [np.mean(errors <= t) for t in tolerances]

    auc = np.trapz(accuracy, tolerances) / tolerances.max()

    plt.figure(figsize=(8, 6))
    plt.plot(tolerances, accuracy, linewidth=2, color='darkorange', label=f'AUC = {auc:.3f}')
    plt.title("REC Curve (Regression Error Characteristic)", fontweight='bold')
    plt.xlabel("Absolute Error Tolerance")
    plt.ylabel("Percentage of Samples within Tolerance")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    print("Starting evaluation...")

    model, tr_embs, tr_tgts, _ = load_model_resources()
    if model is None: print("Model not found."); return

    df_lbl = pd.read_excel(PART_LABEL) if os.path.exists(PART_LABEL) else pd.DataFrame()
    labels = {}
    if not df_lbl.empty:
        cols = [c.lower() for c in df_lbl.columns]
        f = next((c for c in cols if "img" in c or "file" in c), None)
        t = next((c for c in cols if "size" in c or "part" in c), None)
        if f and t:
            for _, r in df_lbl.iterrows(): labels[str(r[f]).strip().lower()] = float(r[t])

    files = sorted([f for f in os.listdir(PART_DIR) if f.lower().endswith((".png", ".jpg", ".tif"))])
    results = []

    print(f"Processing {len(files)} test images...")
    for fn in files:
        img_arr, sc_px = process_image(os.path.join(PART_DIR, fn))
        if img_arr is None or sc_px <= 0: continue

        pred = predict_image(model, img_arr, sc_px, tr_embs, tr_tgts)
        gt = labels.get(fn.lower())

        if gt:
            results.append({"Filename": fn, "GroundTruth": gt, "Prediction": pred})

    if not results: print("No matched ground truth data found."); return

    df = pd.DataFrame(results)
    y_true = df["GroundTruth"].values
    y_pred = df["Prediction"].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-12
    smape = 200 * np.mean(np.abs(y_true - y_pred) / denom)

    spearman, _ = spearmanr(y_true, y_pred)
    pearson, _ = pearsonr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    bias = np.mean(y_pred - y_true)

    metrics = {
        "mae": mae, "rmse": rmse, "mape": mape, "smape": smape,
        "spearman": spearman, "pearson": pearson, "r2": r2, "bias": bias
    }

    report_path = os.path.join(RESULTS_DIR, "Final_Scientific_Report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== EXPERIMENTAL RESULTS ===\n")
        f.write(f"Total Samples: {len(df)}\n\n")
        f.write("--- 1. ERROR METRICS ---\n")
        f.write(f"MAE (Mean Absolute Error): {mae:.4f}\n")
        f.write(f"RMSE (Root Mean Sq Error): {rmse:.4f}\n")
        f.write(f"MAPE (Mean Abs % Error)  : {mape:.2f}%\n")
        f.write(f"sMAPE (Symmetric MAPE)   : {smape:.2f}%\n\n")
        f.write("--- 2. CORRELATION & CONSISTENCY ---\n")
        f.write(f"Spearman rho (Rank)      : {spearman:.4f} (Ranking Ability)\n")
        f.write(f"Pearson r (Linearity)    : {pearson:.4f} (Linear Fit)\n")
        f.write(f"R2 Score                 : {r2:.4f}\n\n")
        f.write("--- 3. BIAS ANALYSIS ---\n")
        f.write(f"Mean Bias (Pred - True)  : {bias:.4f}\n")

        X = y_pred.reshape(-1, 1)
        reg = LinearRegression().fit(X, y_true)
        f.write(f"Calibration Eq: GT = {reg.coef_[0]:.4f} * Pred + {reg.intercept_:.4f}\n")

    print(f"Report saved: {report_path}")

    print("Generating plots...")
    plot_scatter_scientific(y_true, y_pred, metrics, os.path.join(RESULTS_DIR, "Fig1_Scatter_Regression.png"))
    plot_bland_altman(y_true, y_pred, os.path.join(RESULTS_DIR, "Fig2_Bland_Altman.png"))
    plot_error_hist(y_true, y_pred, os.path.join(RESULTS_DIR, "Fig3_Error_Distribution.png"))
    plot_rec_curve(y_true, y_pred, os.path.join(RESULTS_DIR, "Fig4_REC_Curve.png"))

    df.to_csv(os.path.join(RESULTS_DIR, "raw_results.csv"), index=False)
    print(f"All results saved to: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
