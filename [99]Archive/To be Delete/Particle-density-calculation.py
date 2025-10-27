# particle_area_fraction.py
# Requirements: opencv-python, numpy, matplotlib (optional, for plots)

import cv2
import numpy as np
import pathlib
import matplotlib.pyplot as plt

def clahe_unsharp(img_gray, clip=2.0, tile=(8, 8), gk=(9, 9), sigma=10.0,
                  amount=1.5, subtract=0.5):
    """CLAHE + unsharp mask."""
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    x = clahe.apply(img_gray)
    blur = cv2.GaussianBlur(x, gk, sigma)
    unsharp = cv2.addWeighted(x, amount, blur, -subtract, 0)
    return unsharp

def otsu_inverse(unsharp_img):
    """Otsu threshold (inverse) → white = particles (255), black = background (0)."""
    _, binary = cv2.threshold(unsharp_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary

def area_fraction(binary_mask):
    """Fraction of pixels that are foreground (white)."""
    return float((binary_mask > 0).sum()) / binary_mask.size

def process_image(path, show=False, save_mask=True, out_dir="outputs"):
    path = pathlib.Path(path)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    unsharp = clahe_unsharp(img)
    mask = otsu_inverse(unsharp)
    frac = area_fraction(mask)

    if save_mask:
        outp = pathlib.Path(out_dir)
        outp.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(outp / f"{path.stem}_mask.png"), mask)

    if show:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(img, cmap="gray");     ax[0].set_title("Original");         ax[0].axis("off")
        ax[1].imshow(unsharp, cmap="gray"); ax[1].set_title("CLAHE + Unsharp");  ax[1].axis("off")
        ax[2].imshow(mask, cmap="gray");    ax[2].set_title("Binary Mask");      ax[2].axis("off")
        plt.tight_layout(); plt.show()

    return frac

def process_many(paths, show=False, save_mask=True, out_dir="outputs"):
    results = {}
    for p in paths:
        frac = process_image(p, show=show, save_mask=save_mask, out_dir=out_dir)
        results[str(p)] = frac
    return results

if __name__ == "__main__":
    # --- Example usage ---
    # Single image:
    # frac = process_image("/mnt/data/FE26C29D-9758-4B9E-AD89-304FD95208A9.png", show=True)
    # print(f"Area fraction: {frac:.4f}")

    # Batch:
    images = [
        "/mnt/data/90A764DB-61CD-4B8E-AA6D-5A8D12B2EA04.png",
        "/mnt/data/6DF8A39E-AD94-472A-B8F1-99BEB594BA78.png",
        "/mnt/data/FE26C29D-9758-4B9E-AD89-304FD95208A9.png",
        "/mnt/data/19FD379D-7A96-4822-8BF5-76BEC8F050C8.png",
        "/mnt/data/E1A922E5-0E69-4BA5-8502-11628A2822D0.png",
    ]
    results = process_many(images, show=False, save_mask=True)
    for k, v in results.items():
        print(f"{k}: {100*v:.2f}%")