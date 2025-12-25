import cv2
import numpy as np
import os
import pandas as pd
import time
import argparse
from skimage.metrics import structural_similarity as ssim


def load_images(original_path, completed_path):
    original = cv2.imread(original_path)
    completed = cv2.imread(completed_path)

    if original is None:
        raise ValueError(f"Failed to load original image: {original_path}")
    if completed is None:
        raise ValueError(f"Failed to load completed image: {completed_path}")
    if original.shape != completed.shape:
        raise ValueError("Image size mismatch")

    return original, completed


def create_center_mask(image_shape):
    """
    Create a fixed 10% central region mask
    """
    h, w = image_shape[:2]
    ch, cw = int(h * 0.10), int(w * 0.10)

    y0 = (h - ch) // 2
    x0 = (w - cw) // 2

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y0:y0 + ch, x0:x0 + cw] = 255
    return mask


def calculate_ssim(original, completed, mask):
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    completed_gray = cv2.cvtColor(completed, cv2.COLOR_BGR2GRAY)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return float("nan")

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    return ssim(
        original_gray[y0:y1, x0:x1],
        completed_gray[y0:y1, x0:x1],
        data_range=255
    )


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    image_files = [
        f for f in os.listdir(args.original_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
    ]

    print(f"Found {len(image_files)} images")
    print("=" * 60)

    results = []
    start = time.time()

    for img in image_files:
        try:
            orig_path = os.path.join(args.original_dir, img)
            comp_path = os.path.join(args.completed_dir, img)

            if not os.path.exists(comp_path):
                results.append({"Image": img, "SSIM": np.nan})
                continue

            orig, comp = load_images(orig_path, comp_path)
            mask = create_center_mask(orig.shape)
            score = calculate_ssim(orig, comp, mask)

            results.append({"Image": img, "SSIM": score})
            print(f"{img:<40} {score:.4f}")

        except Exception as e:
            print(f"{img:<40} ERROR ({e})")
            results.append({"Image": img, "SSIM": np.nan})

    df = pd.DataFrame(results)
    out_file = os.path.join(args.output_dir, "SSIM_results.xlsx")
    df.to_excel(out_file, index=False)

    print("\nSummary")
    print("=" * 60)
    print(f"Images processed : {len(df)}")
    print(f"Average SSIM     : {df['SSIM'].mean():.4f}")
    print(f"Saved results   : {out_file}")
    print(f"Time elapsed    : {time.time() - start:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute SSIM (fixed 10% center region)")
    parser.add_argument("--original_dir", required=True, help="Directory of original images")
    parser.add_argument("--completed_dir", required=True, help="Directory of completed images")
    parser.add_argument("--output_dir", required=True, help="Output directory")

    args = parser.parse_args()
    main(args)

