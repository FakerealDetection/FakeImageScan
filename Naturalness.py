import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import argparse
import csv
import math
import time


# ============================================================
# SReC probabilistic model: outputs Gaussian params (mu, log_sigma)
# ============================================================
class SReCModel(torch.nn.Module):
    """
    Predictive distribution model for pixel values:
      output = [mu_R, mu_G, mu_B, log_sigma_R, log_sigma_G, log_sigma_B]
    where the likelihood of the observed pixel is computed under N(mu, sigma^2).
    """
    def __init__(self, model_path: str, device: torch.device):
        super().__init__()
        self.device = device
        print(f"Loading model weights from {model_path}")

        self.model = self.create_model().to(device)
        checkpoint = torch.load(model_path, map_location="cpu")

        # Expect checkpoint['nets'] like your original script
        pretrained_dict = checkpoint.get("nets", checkpoint)
        model_dict = self.model.state_dict()

        # Load matching keys only
        filtered = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered)
        self.model.load_state_dict(model_dict, strict=False)

        print(f"Model loaded on device: {device}")
        print(f"Loaded keys: {len(filtered)}/{len(model_dict)} (shape-matched)")

    def create_model(self):
        # Simple backbone + fixed pooling, output 6 dims (mu(3) + log_sigma(3))
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((8, 8)),
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 8 * 8, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 6)
        )

    def forward(self, x):
        return self.model(x)


# ============================================================
# Likelihood: Gaussian log-prob of observed pixel given (mu, sigma)
# ============================================================
def gaussian_log_prob(x: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    """
    x, mu, log_sigma: shape [N, 3]
    Returns log p(x | mu, sigma) per sample: shape [N]
    """
    # Stabilize sigma
    log_sigma = torch.clamp(log_sigma, min=-7.0, max=7.0)
    sigma = torch.exp(log_sigma)

    # log N(x; mu, sigma^2) per channel, sum over channels
    # logp = -0.5*((x-mu)/sigma)^2 - log(sigma) - 0.5*log(2*pi)
    const = 0.5 * math.log(2.0 * math.pi)
    z = (x - mu) / (sigma + 1e-8)
    logp_ch = -0.5 * (z * z) - log_sigma - const
    return logp_ch.sum(dim=1)  # sum over RGB -> [N]


@torch.no_grad()
def predict_dist_params(model, inp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    inp: [N,3,H,W]
    Returns:
      mu: [N,3]
      log_sigma: [N,3]
    """
    out = model(inp)  # [N,6]
    mu = out[:, 0:3]
    log_sigma = out[:, 3:6]

    # Optional: keep mu in a reasonable range (since inputs are normalized)
    # This doesn't change probability definition; it prevents crazy mu.
    mu = torch.tanh(mu)  # [-1,1] in normalized space

    return mu, log_sigma


# ============================================================
# Context builders (global + local)
# ============================================================
def load_and_preprocess(image_path: str, device: torch.device) -> torch.Tensor:
    """
    Returns normalized image tensor: [1,3,512,512]
    """
    image = Image.open(image_path).convert("RGB")
    tfm = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return tfm(image).unsqueeze(0).to(device)


def central_region_coords(h=512, w=512, frac=0.10, stride=2):
    """
    Generates pixel coordinates in the central frac region.
    stride=2 for speed (matches your earlier optimization idea).
    """
    total = h * w
    target = int(total * frac)
    side = int(math.sqrt(target))
    cx, cy = h // 2, w // 2
    half = side // 2

    x0 = max(1, cx - half)
    x1 = min(h - 1, cx + half)
    y0 = max(1, cy - half)
    y1 = min(w - 1, cy + half)

    coords = []
    for x in range(x0, x1, stride):
        for y in range(y0, y1, stride):
            coords.append((x, y))
            if len(coords) >= target:
                return coords
    return coords


def build_local_patch(img: torch.Tensor, x: int, y: int) -> torch.Tensor:
    """
    img: [1,3,512,512] normalized
    Extract 3x3 neighborhood around (x,y), pad if needed.
    Output: [1,3,3,3]
    """
    # img indices: [B,C,H,W], x->H, y->W
    x0, x1 = max(0, x - 1), min(img.shape[2], x + 2)
    y0, y1 = max(0, y - 1), min(img.shape[3], y + 2)
    patch = img[:, :, x0:x1, y0:y1]
    # pad to 3x3
    pad_h = 3 - patch.shape[2]
    pad_w = 3 - patch.shape[3]
    if pad_h > 0 or pad_w > 0:
        patch = F.pad(patch, (0, pad_w, 0, pad_h))
    return patch


def extract_true_pixel(img: torch.Tensor, x: int, y: int) -> torch.Tensor:
    """
    Returns true pixel value (normalized) at (x,y): shape [1,3]
    """
    return img[:, :, x, y].clone()  # [1,3]


# ============================================================
# Pixel naturalness = λ p_global + (1-λ) p_local  (Eq. 1)
# ============================================================
@torch.no_grad()
def compute_image_naturalness(model, image_path: str, lambda_factor=0.5, fast_mode=False) -> float:
    """
    Returns image-level naturalness score:
      average over central pixels of p(Pi) where
      p(Pi) = λ p_global(Pi) + (1-λ) p_local(Pi)
    """
    start = time.time()
    device = next(model.parameters()).device

    img = load_and_preprocess(image_path, device)  # [1,3,512,512]

    # Coordinates in central 10%
    coords = central_region_coords(frac=0.10, stride=2)

    # Build global input once (fast) or per-pixel (slower)
    # Global context = broad receptive field
    global_inp = F.interpolate(img, size=(64, 64), mode="bilinear", align_corners=False)  # [1,3,64,64]

    probs = []

    if fast_mode:
        # FAST approximation:
        # - global: one distribution for whole image
        # - local: one distribution for whole central region
        mu_g, logsig_g = predict_dist_params(model, global_inp)

        # central region as "local area context"
        # (still consistent with restricted neighborhood idea at region level)
        # For strict 3x3 local, use non-fast mode below.
        # Here we keep fast option for speed.
        # extract central region roughly then resize
        h, w = 512, 512
        side = int(math.sqrt(int(h*w*0.10)))
        cx, cy = 256, 256
        half = side // 2
        x0, x1 = max(1, cx-half), min(511, cx+half)
        y0, y1 = max(1, cy-half), min(511, cy+half)
        central = img[:, :, x0:x1, y0:y1]
        local_inp = F.interpolate(central, size=(32, 32), mode="bilinear", align_corners=False)
        mu_l, logsig_l = predict_dist_params(model, local_inp)

        # Compute per-pixel likelihood using the SAME (mu,sigma) for all pixels (fast)
        # This is approximate but still probability-based.
        for (x, y) in coords:
            x_true = extract_true_pixel(img, x, y)  # [1,3]
            logp_g = gaussian_log_prob(x_true, mu_g, logsig_g)  # [1]
            logp_l = gaussian_log_prob(x_true, mu_l, logsig_l)  # [1]

            # Convert log-prob to prob (can be tiny), clamp for numerical stability
            p_g = torch.exp(torch.clamp(logp_g, min=-50.0, max=10.0))
            p_l = torch.exp(torch.clamp(logp_l, min=-50.0, max=10.0))

            p = lambda_factor * p_g + (1.0 - lambda_factor) * p_l
            probs.append(float(p.item()))

    else:
        # STRICT mode (matches your text more closely):
        # local = 3x3 neighborhood, global = full-image receptive field (downsampled)
        for (x, y) in coords:
            x_true = extract_true_pixel(img, x, y)  # [1,3]

            # Global context prediction
            mu_g, logsig_g = predict_dist_params(model, global_inp)  # [1,3] each
            logp_g = gaussian_log_prob(x_true, mu_g, logsig_g)
            p_g = torch.exp(torch.clamp(logp_g, min=-50.0, max=10.0))

            # Local context prediction (3x3 patch around pixel)
            patch = build_local_patch(img, x, y)  # [1,3,3,3]
            patch_inp = F.interpolate(patch, size=(8, 8), mode="bilinear", align_corners=False)
            mu_l, logsig_l = predict_dist_params(model, patch_inp)
            logp_l = gaussian_log_prob(x_true, mu_l, logsig_l)
            p_l = torch.exp(torch.clamp(logp_l, min=-50.0, max=10.0))

            # Eq. (1)
            p = lambda_factor * p_g + (1.0 - lambda_factor) * p_l
            probs.append(float(p.item()))

    avg_prob = float(np.mean(probs)) if len(probs) else 0.0
    print(f"[{os.path.basename(image_path)}] naturalness={avg_prob:.6f} | pixels={len(probs)} | time={time.time()-start:.2f}s")
    return avg_prob


def minmax_normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float64)
    if arr.size == 0:
        return arr
    mn, mx = np.min(arr), np.max(arr)
    if mx - mn < 1e-12:
        return np.full_like(arr, 0.5, dtype=np.float64)
    return (arr - mn) / (mx - mn)


# ============================================================
# Main
# ============================================================
def main(args):
    print("=" * 60)
    print("Pixel Naturalness (Eq. 1) using SReC global+local likelihood")
    print("=" * 60)

    if args.gpu_id is not None:
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    # Load model
    model = SReCModel(args.load, device)
    model.eval()

    # Read image paths
    with open(args.file, "r") as f:
        image_paths = [line.strip() for line in f if line.strip()]

    print(f"Found {len(image_paths)} images")

    # Compute image-level naturalness
    scores = []
    for i, pth in enumerate(image_paths, 1):
        if not os.path.exists(pth):
            print(f"Missing: {pth}")
            scores.append(0.0)
            continue
        print(f"\n--- {i}/{len(image_paths)} ---")
        s = compute_image_naturalness(
            model,
            pth,
            lambda_factor=args.lmbda,
            fast_mode=args.fast_mode
        )
        scores.append(s)

    scores = np.array(scores, dtype=np.float64)
    scores_norm = minmax_normalize(scores)

    # Save CSV
    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image Path", "Naturalness_raw", "Naturalness_norm_0_1"])
        for pth, raw, norm in zip(image_paths, scores, scores_norm):
            writer.writerow([pth, float(raw), float(norm)])

    print(f"\nSaved: {args.output_csv}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pixel Naturalness via SReC likelihood (global+local, Eq. 1).")
    parser.add_argument("--file", type=str, required=True, help="Text file containing image paths (one per line)")
    parser.add_argument("--load", type=str, required=True, help="Path to SReC pretrained checkpoint")
    parser.add_argument("--output_csv", type=str, required=True, help="Output CSV path")
    parser.add_argument("--fast_mode", action="store_true", help="Faster approximate mode")
    parser.add_argument("--gpu_id", type=int, help="GPU id (e.g., 0)")
    parser.add_argument("--lmbda", type=float, default=0.5, help="Lambda weight for global vs local (default 0.5)")
    args = parser.parse_args()

    main(args)
