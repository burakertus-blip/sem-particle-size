import os
import cv2
import numpy as np
import random
import csv
import math

try:
    import torch
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

IMG_SIZE = 512
SCALE_BAR_PX = 80

OUTPUT_DIR = "synthetic_sem_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUTPUT_DIR, "labels.csv")

ratios = np.round(np.arange(0.025, 2.200 + 1e-9, 0.025), 3)

fills = [0.10, 0.25, 0.50]
shapes = ["circle", "rounded_square"]

num_random = 1
ROT_MAX_DEG = 15.0
img_counter = 1

N_MIN_LO, N_MIN_HI = 4, 6
N_MAX_LO, N_MAX_HI = 400, 600
N_MAX_DEFAULT = 520

def max_overlap_ratio_from_fill(fill_fraction: float) -> float:
    return float(np.clip(0.06 + 0.36 * fill_fraction, 0.08, 0.30))

BG_BLUR_K = 3
EFFECT_BLUR_K = 3

if TORCH_OK and torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print("Torch:", TORCH_OK, "| DEVICE:", DEVICE)
print(f"Ratios: {len(ratios)} | Fills: {len(fills)} | Shapes: {len(shapes)}")
print(f"Estimated images: {len(ratios)*len(fills)*len(shapes)*num_random*2}")

def sem_background():
    if TORCH_OK and DEVICE == "cuda":
        bg = torch.normal(mean=70.0, std=10.0, size=(IMG_SIZE, IMG_SIZE), device=DEVICE).clamp(0, 255)
        bg = bg.to(torch.float32)

        yy, xx = torch.meshgrid(torch.linspace(-1, 1, IMG_SIZE, device=DEVICE),
                                torch.linspace(-1, 1, IMG_SIZE, device=DEVICE),
                                indexing="ij")
        ax = float(random.uniform(-12, 12))
        ay = float(random.uniform(-12, 12))
        bg = bg + (ax * xx + ay * yy)

        bg = bg.unsqueeze(0).unsqueeze(0)
        k = 3
        kernel = torch.ones((1, 1, k, k), device=DEVICE) / (k*k)
        bg = torch.nn.functional.conv2d(bg, kernel, padding=k//2).squeeze()

        return bg.clamp(0, 255).to(torch.uint8).cpu().numpy()
    else:
        bg = np.random.normal(70, 10, (IMG_SIZE, IMG_SIZE)).astype(np.float32)

        yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE].astype(np.float32)
        xx = (xx - IMG_SIZE/2) / (IMG_SIZE/2)
        yy = (yy - IMG_SIZE/2) / (IMG_SIZE/2)
        ax = random.uniform(-12, 12)
        ay = random.uniform(-12, 12)
        bg += (ax * xx + ay * yy)

        bg = np.clip(bg, 0, 255).astype(np.uint8)
        if BG_BLUR_K and BG_BLUR_K >= 3:
            bg = cv2.GaussianBlur(bg, (BG_BLUR_K, BG_BLUR_K), 0)
        return bg

def add_sem_effects(img):
    if TORCH_OK and DEVICE == "cuda":
        t = torch.from_numpy(img).to(DEVICE).to(torch.float32)

        noise = torch.normal(mean=0.0, std=6.0, size=t.shape, device=DEVICE)
        t = (t + noise).clamp(0, 255)

        if EFFECT_BLUR_K and EFFECT_BLUR_K >= 3:
            t = t.unsqueeze(0).unsqueeze(0)
            k = 3
            kernel = torch.ones((1, 1, k, k), device=DEVICE) / (k*k)
            t = torch.nn.functional.conv2d(t, kernel, padding=k//2).squeeze()

        t = (t / 255.0) ** 1.08 * 255.0
        return t.clamp(0, 255).to(torch.uint8).cpu().numpy()
    else:
        x = img.astype(np.float32)
        noise = np.random.normal(0, 6, img.shape).astype(np.float32)
        x = np.clip(x + noise, 0, 255)

        if EFFECT_BLUR_K and EFFECT_BLUR_K >= 3:
            x = cv2.GaussianBlur(x, (EFFECT_BLUR_K, EFFECT_BLUR_K), 0)

        x = (x / 255.0) ** 1.08 * 255.0
        return np.clip(x, 0, 255).astype(np.uint8)

def rotate_patch(patch: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = patch.shape
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    out = cv2.warpAffine(patch, M, (w, h), flags=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out

def paste_patch(dst: np.ndarray, patch: np.ndarray, x: int, y: int):
    h, w = patch.shape
    roi = dst[y:y+h, x:x+w]
    np.maximum(roi, patch, out=roi)

def draw_rounded_square_patch(d: int) -> np.ndarray:
    d = max(8, int(d))
    pad = 10
    size = d + 2 * pad
    patch = np.zeros((size, size), dtype=np.uint8)

    rad = max(2, int(0.18 * d))
    rad = min(rad, d // 2 - 1)

    x0, y0 = pad, pad
    cv2.rectangle(patch, (x0 + rad, y0), (x0 + d - rad, y0 + d), 255, -1)
    cv2.rectangle(patch, (x0, y0 + rad), (x0 + d, y0 + d - rad), 255, -1)
    cv2.circle(patch, (x0 + rad, y0 + rad), rad, 255, -1)
    cv2.circle(patch, (x0 + d - rad, y0 + rad), rad, 255, -1)
    cv2.circle(patch, (x0 + rad, y0 + d - rad), rad, 255, -1)
    cv2.circle(patch, (x0 + d - rad, y0 + d - rad), rad, 255, -1)
    return patch

def draw_circle_patch(d: int) -> np.ndarray:
    d = max(6, int(d))
    r = max(3, d // 2)
    pad = 8
    size = 2 * r + 2 * pad
    patch = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(patch, (size // 2, size // 2), r, 255, -1)
    return patch

def stamp_shape_with_shading(img: np.ndarray, mask_full: np.ndarray):
    sel = (mask_full > 0)
    if not np.any(sel):
        return

    ys, xs = np.where(sel)
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    roi_mask = mask_full[y1:y2+1, x1:x2+1]
    roi_img  = img[y1:y2+1, x1:x2+1].astype(np.float32)

    m = (roi_mask > 0).astype(np.uint8)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    if dist.max() <= 1e-6:
        return
    dist_n = dist / (dist.max() + 1e-6)

    h, w = roi_mask.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx = (xx - (w-1)/2) / ((w-1)/2 + 1e-6)
    yy = (yy - (h-1)/2) / ((h-1)/2 + 1e-6)

    theta = random.uniform(0, 2*np.pi)
    vx, vy = math.cos(theta), math.sin(theta)
    dir_term = (vx * xx + vy * yy)
    dir_n = (dir_term + 1.0) / 2.0

    base_int = float(np.clip(random.gauss(205, 18), 140, 240))
    radial = 0.70 + 0.30 * dist_n
    directional = 0.88 + 0.24 * dir_n
    target = np.clip(base_int * radial * directional, 0, 255)

    alpha = 0.70
    sel_roi = (roi_mask > 0)
    roi_img[sel_roi] = (1 - alpha) * roi_img[sel_roi] + alpha * target[sel_roi]

    img[y1:y2+1, x1:x2+1] = np.clip(roi_img, 0, 255).astype(np.uint8)

def add_edge_highlight(img: np.ndarray, occ_mask: np.ndarray):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edge = cv2.morphologyEx(occ_mask, cv2.MORPH_GRADIENT, k)
    edge = cv2.dilate(edge, k, iterations=1)

    boost = float(np.clip(random.gauss(22, 4), 10, 32))
    out = img.astype(np.float32)
    out[edge > 0] = np.clip(out[edge > 0] + boost, 0, 255)
    return out.astype(np.uint8)

def choose_n_limits():
    nmin = random.randint(N_MIN_LO, N_MIN_HI)
    nmax = random.randint(N_MAX_LO, N_MAX_HI)
    return nmin, min(nmax, N_MAX_DEFAULT)

def place_particles_true_fill(img: np.ndarray, shape: str, ratio: float, fill_fraction: float):
    diameter_px = int(round(ratio * SCALE_BAR_PX))
    diameter_px = max(6, diameter_px)

    if shape == "circle":
        base_patch = draw_circle_patch(diameter_px)
    else:
        base_patch = draw_rounded_square_patch(diameter_px)

    max_overlap_ratio = max_overlap_ratio_from_fill(fill_fraction)

    occ = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    current_area = 0
    target_area = int(fill_fraction * IMG_SIZE * IMG_SIZE)

    nmin, nmax = choose_n_limits()
    max_particles_hard = nmax

    max_tries_per_particle = 280
    placed_count = 0
    tried_particles = 0

    while current_area < target_area and placed_count < max_particles_hard:
        tried_particles += 1
        if tried_particles > max_particles_hard * 12:
            break

        if shape == "rounded_square":
            ang = random.uniform(-ROT_MAX_DEG, ROT_MAX_DEG)
            patch = rotate_patch(base_patch, ang)
        else:
            patch = base_patch

        ys, xs = np.where(patch > 0)
        if len(xs) == 0:
            continue

        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        patch_c = patch[y1:y2+1, x1:x2+1]
        h, w = patch_c.shape

        if h >= IMG_SIZE or w >= IMG_SIZE:
            x = max(0, (IMG_SIZE - w) // 2)
            y = max(0, (IMG_SIZE - h) // 2)
        else:
            x = random.randint(0, IMG_SIZE - w - 1)
            y = random.randint(0, IMG_SIZE - h - 1)

        temp = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        paste_patch(temp, patch_c, x, y)

        new_area = int(np.count_nonzero(temp))
        if new_area == 0:
            continue

        overlap = int(np.count_nonzero((temp > 0) & (occ > 0)))
        overlap_ratio = overlap / float(new_area)

        if overlap_ratio <= max_overlap_ratio:
            added = new_area - overlap
            if added <= 0:
                continue

            stamp_shape_with_shading(img, temp)
            occ = cv2.bitwise_or(occ, temp)

            current_area += added
            placed_count += 1

    if placed_count < nmin:
        extra_tries = 0
        while placed_count < nmin and extra_tries < 2000:
            extra_tries += 1
            if shape == "rounded_square":
                ang = random.uniform(-ROT_MAX_DEG, ROT_MAX_DEG)
                patch = rotate_patch(base_patch, ang)
            else:
                patch = base_patch

            ys, xs = np.where(patch > 0)
            if len(xs) == 0:
                continue

            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            patch_c = patch[y1:y2+1, x1:x2+1]
            h, w = patch_c.shape

            x = random.randint(0, IMG_SIZE - w - 1)
            y = random.randint(0, IMG_SIZE - h - 1)

            temp = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            paste_patch(temp, patch_c, x, y)

            new_area = int(np.count_nonzero(temp))
            overlap = int(np.count_nonzero((temp > 0) & (occ > 0)))
            overlap_ratio = overlap / float(new_area + 1e-9)

            if overlap_ratio <= (max_overlap_ratio + 0.08):
                stamp_shape_with_shading(img, temp)
                occ = cv2.bitwise_or(occ, temp)
                placed_count += 1

    achieved_fill = float(np.count_nonzero(occ)) / float(IMG_SIZE * IMG_SIZE)

    return occ, placed_count, achieved_fill, target_area, current_area

def save_with_label(writer, img, ratio, fill_fraction, shape, variant, n_placed, achieved_fill):
    global img_counter
    filename = f"img{img_counter}.png"
    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), img)
    writer.writerow({
        "image_name": filename,
        "particle_size_ratio": f"{ratio:.3f}",
        "fill_fraction": f"{fill_fraction:.2f}",
        "shape": shape,
        "variant": variant,
        "n_placed": int(n_placed),
        "achieved_fill": f"{achieved_fill:.4f}",
    })
    img_counter += 1

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["image_name", "particle_size_ratio", "fill_fraction", "shape", "variant", "n_placed", "achieved_fill"]
    )
    writer.writeheader()

    for ratio in ratios:
        for fill_fraction in fills:
            for shape in shapes:
                for _ in range(num_random):
                    img = sem_background()

                    occ, n_placed, achieved_fill, _, _ = place_particles_true_fill(
                        img, shape, ratio, fill_fraction
                    )

                    img = add_edge_highlight(img, occ)
                    img = add_sem_effects(img)

                    save_with_label(writer, img, ratio, fill_fraction, shape, "orig", n_placed, achieved_fill)

                    img_rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    save_with_label(writer, img_rot, ratio, fill_fraction, shape, "rot90", n_placed, achieved_fill)

print(f"Done. Output directory: {OUTPUT_DIR}")
print(f"Total images generated: {img_counter-1}")
print(f"CSV: {CSV_PATH}")
print(f"Ratios: {len(ratios)} | Fills: {len(fills)} | Shapes: {len(shapes)}")
print(f"Estimated total: {len(ratios)*len(fills)*len(shapes)*num_random*2}")
