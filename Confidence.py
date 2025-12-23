import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


# -------------------------
# Utilities
# -------------------------
def load_image_bgr(path: str) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not load image {path}")
    return img

def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def pad_to_multiple(tensor, multiple=8):
    """Pad tensor so H,W divisible by multiple (PSPNet commonly needs /8 aligned)."""
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    if pad_h == 0 and pad_w == 0:
        return tensor, (0, 0, 0, 0)

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padded = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")
    return padded, (pad_top, pad_bottom, pad_left, pad_right)

def unpad_tensor(tensor, padding):
    pad_top, pad_bottom, pad_left, pad_right = padding
    if padding == (0, 0, 0, 0):
        return tensor
    _, _, h, w = tensor.shape
    return tensor[:, :, pad_top:h - pad_bottom, pad_left:w - pad_right]


# -------------------------
# PSPNet
# -------------------------
def load_pspnet(device: str):
    from segmentation_models_pytorch import PSPNet
    model = PSPNet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        classes=21,
        activation=None
    )
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def compute_pspnet_confidence(model, img_bgr, prep, device: str) -> float:
    """
    Returns a confidence score in [0, 1]:
    mean over pixels of max softmax probability.
    """
    pil_img = bgr_to_pil(img_bgr)
    x = prep(pil_img).unsqueeze(0).to(device)  # [1,3,H,W]

    x, padding = pad_to_multiple(x, multiple=8)

    out = model(x)

    # segmentation_models_pytorch PSPNet may return:
    # - torch.Tensor [B,C,H,W]
    # - tuple/list where first is [B,C,H,W]
    if isinstance(out, (tuple, list)):
        logits = out[0]
    else:
        logits = out

    logits = unpad_tensor(logits, padding)  # [1,C,H,W]
    probs = torch.softmax(logits, dim=1)    # [1,C,H,W]
    max_prob = probs.max(dim=1).values      # [1,H,W]
    conf = float(max_prob.mean().item())    # scalar

    # Safety clamp (should already be in [0,1])
    conf = max(0.0, min(1.0, conf))
    return conf


def minmax_normalize(values):
    """Min-max normalize list/array to [0,1], keeping NaNs as NaN."""
    arr = np.array(values, dtype=np.float64)
    mask = np.isfinite(arr)
    if not np.any(mask):
        return [np.nan] * len(arr)

    vmin = np.min(arr[mask])
    vmax = np.max(arr[mask])
    if abs(vmax - vmin) < 1e-12:
        # all same -> normalize to 0.5 (or 0). Here we choose 0.5.
        out = np.full_like(arr, np.nan, dtype=np.float64)
        out[mask] = 0.5
        return out.tolist()

    out = np.full_like(arr, np.nan, dtype=np.float64)
    out[mask] = (arr[mask] - vmin) / (vmax - vmin)
    return out.tolist()


def main():
    ap = argparse.ArgumentParser(description="Compute PSPNet confidence for images (0..1) + min-max normalization (0..1).")
    ap.add_argument("--input_dir", required=True, help="Folder with images (recursive)")
    ap.add_argument("--output_csv", required=True, help="Where to write CSV")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Load model
    print("Loading PSPNet...")
    try:
        model = load_pspnet(device)
        print("✓ PSPNet loaded")
    except Exception as e:
        print(f"✗ Failed to load PSPNet: {e}")
        return

    # ImageNet normalization (this is the required normalization)
    prep = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Collect image paths
    paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.tif", "*.tiff"]:
        paths.extend(Path(args.input_dir).rglob(ext))
        paths.extend(Path(args.input_dir).rglob(ext.upper()))
    paths = sorted([str(p) for p in paths])

    print(f"Found {len(paths)} images")

    rows = []
    for p in tqdm(paths, desc="Computing confidence"):
        try:
            img_bgr = load_image_bgr(p)
            conf = compute_pspnet_confidence(model, img_bgr, prep, device)
            rows.append({"path": p, "PSPNet_confidence": conf})
        except Exception as e:
            rows.append({"path": p, "PSPNet_confidence": np.nan, "error": str(e)})

    if not rows:
        print("No images processed.")
        return

    df = pd.DataFrame(rows)
    df["confidence_norm"] = minmax_normalize(df["PSPNet_confidence"].tolist())

    # Ensure output folder exists
    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df.to_csv(args.output_csv, index=False)
    print(f"Saved: {args.output_csv}")
    print("Columns:", df.columns.tolist())


if __name__ == "__main__":
    main()
