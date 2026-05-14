"""Quick visual check: blend a sky mask over its source image and save the result."""
import argparse
from pathlib import Path
import cv2
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--scene_dir", required=True)
p.add_argument("--frame", type=int, default=0)
p.add_argument("--cam", type=int, default=0)
p.add_argument("--out", required=True)
a = p.parse_args()

img = cv2.imread(str(Path(a.scene_dir) / "images" / f"{a.frame:03d}_{a.cam}.jpg"))
mask = cv2.imread(str(Path(a.scene_dir) / "sky_masks" / f"{a.frame:03d}_{a.cam}.png"),
                  cv2.IMREAD_GRAYSCALE)
overlay = img.copy()
overlay[mask > 0] = (0, 255, 255)  # yellow where mask says "sky"
blend = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
cv2.imwrite(a.out, blend)
sky_frac = (mask > 0).mean()
print(f"sky pixel fraction: {sky_frac*100:.1f}%  -> {a.out}")
