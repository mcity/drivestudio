"""Project saved lidar points into a camera image and save an overlay.

Reads the on-disk Waymo-format outputs of preprocess_mcity.py and reverses
the loader's camera-frame transform to project lidar (in ego frame) back into
each camera image. If calibration is correct, lidar points should align with
image edges.

Usage:
  python tools/verify_mcity_overlay.py \
    --scene_dir /scratch/.../mcity/processed/training/000 \
    --frame 0 \
    --out_dir /scratch/.../mcity/verify
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np


OPENCV2DATASET = np.array([
    [0, 0, 1, 0],
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1],
], dtype=np.float64)


def project_one(scene_dir: Path, frame: int, cam: int, out_dir: Path):
    img = cv2.imread(str(scene_dir / "images" / f"{frame:03d}_{cam}.jpg"))
    if img is None:
        print(f"[skip] cam{cam}: no image"); return
    h, w = img.shape[:2]

    intr = np.loadtxt(scene_dir / "intrinsics" / f"{cam}.txt")
    fx, fy, cx, cy = intr[0:4]
    k1, k2, p1, p2, k3 = intr[4:9]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

    # Undistort with the same K, exactly as the loader does (cv2.undistort with
    # undistort=True in the mcity config). The model consumes this undistorted
    # image under a pinhole K, so we project with K and NO distortion below.
    img = cv2.undistort(img, K, dist)

    extr_file = np.loadtxt(scene_dir / "extrinsics" / f"{cam}.txt")
    # loader: cam_to_ego = file @ OPENCV2DATASET -> T_opencv_cam_to_ego
    T_cam_to_ego = extr_file @ OPENCV2DATASET
    T_ego_to_cam = np.linalg.inv(T_cam_to_ego)

    raw = np.fromfile(scene_dir / "lidar" / f"{frame:03d}.bin", dtype=np.float32)
    pts_ego = raw.reshape(-1, 14)[:, 3:6].astype(np.float64)

    homo = np.concatenate([pts_ego, np.ones((len(pts_ego), 1))], axis=1)
    pts_cam = (T_ego_to_cam @ homo.T).T[:, :3]
    in_front = pts_cam[:, 2] > 0.1
    pts_cam = pts_cam[in_front]
    depths = pts_cam[:, 2]

    # pinhole projection onto the undistorted image (distortion already removed)
    uv = (K @ pts_cam.T).T
    uv = uv[:, :2] / uv[:, 2:3]

    valid = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uv = uv[valid]; depths = depths[valid]

    d_min, d_max = max(depths.min(), 1.0), min(depths.max(), 60.0)
    norm = np.clip((depths - d_min) / max(d_max - d_min, 1e-6), 0, 1)
    colors = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET).reshape(-1, 3)

    out = img.copy()
    for (u, v), c in zip(uv.astype(int), colors):
        cv2.circle(out, (u, v), 2, (int(c[0]), int(c[1]), int(c[2])), -1)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"overlay_f{frame:03d}_cam{cam}.jpg"
    cv2.imwrite(str(out_path), out)
    print(f"[ok] cam{cam}: {len(uv)} projected pts -> {out_path}")


def plot_trajectory(scene_dir: Path, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pose_files = sorted((scene_dir / "ego_pose").glob("*.txt"))
    xs, ys = [], []
    for p in pose_files:
        T = np.loadtxt(p)
        xs.append(T[0, 3]); ys.append(T[1, 3])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(xs, ys, marker=".", linewidth=1)
    ax.set_aspect("equal"); ax.set_title(f"ego trajectory ({len(xs)} frames)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "trajectory.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] trajectory.png ({len(xs)} frames, span x: {max(xs)-min(xs):.1f}m, y: {max(ys)-min(ys):.1f}m)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    scene_dir = Path(args.scene_dir)
    out_dir = Path(args.out_dir)
    for c in range(6):
        project_one(scene_dir, args.frame, c, out_dir)
    plot_trajectory(scene_dir, out_dir)


if __name__ == "__main__":
    main()
