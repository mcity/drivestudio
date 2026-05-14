"""Convert an Mcity ROS2 mcap bag into Waymo-format drivestudio scene.

Mandatory outputs per frame in {out_dir}:
  images/{f:03d}_{c}.jpg         (c in 0..5 = arenacam1..6)
  lidar/{f:03d}.bin              (N x 14 float32, Waymo schema)
  ego_pose/{f:03d}.txt           (4x4 matrix, oxts_link -> map)
  intrinsics/{c}.txt             (9 scalars: fx, fy, cx, cy, k1, k2, p1, p2, k3)
  extrinsics/{c}.txt             (4x4 matrix, cam -> ego, in Waymo cam convention)
  frame_info.json                (stub metadata)

LiDAR is taken from /rslidar_front_points only (v1). Other lidars are skipped
because their per-physical-lidar-to-combined transforms are not in the
calibration. Camera-to-ego is composed via:
    T_cam_to_ego = T_combined_to_imu @ inv(T_lidar_to_cam)
under the assumption that the calibration's `front_lidar`/`left_lidar`/
`right_lidar` source frames are all expressed in the
`rslidar_combined_aligned_fixed` frame. Verify with project_overlay.py.

Usage:
  python tools/preprocess_mcity.py \
    --bag_path /scratch/.../may8-2026-downtown-p2_0.mcap \
    --calib_root /home/billhong/drivestudio/calibration_files \
    --out_dir /scratch/.../mcity/processed/training/000 \
    --start_s 180 --end_s 200 --hz 10
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.reader import read_ros2_messages


OPENCV2DATASET = np.array([
    [0, 0, 1, 0],
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1],
], dtype=np.float64)

ARENACAM_TOPICS = [f"/arenacam{i+1}/images" for i in range(6)]
LIDAR_TOPIC = "/rslidar_front_points"
TF_TOPIC = "/tf"

# Cam idx 0..5 -> filename in lidar_cam_joint_extrinsics/
LIDAR_TO_CAM_FILES = {
    0: "front_lidar-to-cam1-extrinsic.json",
    1: "front_lidar-to-cam2-extrinsic.json",
    2: "left_lidar-to-cam3-extrinsic.json",
    3: "right_lidar-to-cam4-extrinsic.json",
    4: "left_lidar-to-cam5-extrinsic.json",
    5: "right_lidar-to-cam6-extrinsic.json",
}


def load_intrinsic(path):
    with open(path) as f:
        d = json.load(f)
    inner = d[next(iter(d))]["param"]
    K = np.array(inner["cam_K"]["data"], dtype=np.float64)
    dist = np.array(inner["cam_dist"]["data"], dtype=np.float64).flatten()
    w, h = inner["img_dist_w"], inner["img_dist_h"]
    return K, dist, (w, h)


def load_extrinsic(path):
    with open(path) as f:
        d = json.load(f)
    inner = d[next(iter(d))]["param"]
    T = np.array(inner["sensor_calib"]["data"], dtype=np.float64)
    return T


def quat_to_R(qx, qy, qz, qw):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def transform_to_matrix(tr):
    t = tr.translation
    q = tr.rotation
    R = quat_to_R(q.x, q.y, q.z, q.w)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def stamp_to_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def decode_image(msg):
    enc = msg.encoding.lower()
    h, w = msg.height, msg.width
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    if enc in ("bgr8",):
        return raw.reshape(h, w, 3)
    if enc in ("rgb8",):
        return cv2.cvtColor(raw.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    if enc in ("mono8",):
        return cv2.cvtColor(raw.reshape(h, w), cv2.COLOR_GRAY2BGR)
    # OpenCV's Bayer code naming is shifted by one row from ROS's.
    # ROS bayer_rggb8 (RGGB sensor pattern) -> cv2.COLOR_BAYER_BG2BGR, etc.
    bayer_codes = {
        "bayer_rggb8": cv2.COLOR_BAYER_BG2BGR,
        "bayer_bggr8": cv2.COLOR_BAYER_RG2BGR,
        "bayer_gbrg8": cv2.COLOR_BAYER_GR2BGR,
        "bayer_grbg8": cv2.COLOR_BAYER_GB2BGR,
    }
    if enc in bayer_codes:
        return cv2.cvtColor(raw.reshape(h, w), bayer_codes[enc])
    raise NotImplementedError(f"unsupported image encoding: {enc}")


def decode_pointcloud2(msg):
    """Return (xyz [N,3] float32, intensity [N] float32)."""
    offsets = {}
    dtypes = {}
    dtype_map = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
                 5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64}
    for f in msg.fields:
        offsets[f.name] = f.offset
        dtypes[f.name] = dtype_map.get(f.datatype, np.float32)

    point_step = msg.point_step
    n = len(msg.data) // point_step
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, point_step)

    def col(name, default=None):
        if name not in offsets:
            if default is None:
                return None
            return np.full(n, default, dtype=np.float32)
        dt = dtypes[name]
        off = offsets[name]
        sz = np.dtype(dt).itemsize
        return raw[:, off:off + sz].copy().view(dt).flatten().astype(np.float32)

    x = col("x"); y = col("y"); z = col("z")
    if x is None or y is None or z is None:
        raise ValueError("PointCloud2 missing x/y/z")
    intensity = col("intensity", default=0.0)
    xyz = np.stack([x, y, z], axis=1).astype(np.float32)

    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    intensity = intensity[finite]
    return xyz, intensity


def transform_points(T, pts):
    homo = np.concatenate([pts, np.ones((len(pts), 1), dtype=pts.dtype)], axis=1)
    return (T @ homo.T).T[:, :3]


def get_bag_time_range(bag_path):
    with open(bag_path, "rb") as f:
        reader = make_reader(f)
        s = reader.get_summary()
        return s.statistics.message_start_time, s.statistics.message_end_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag_path", required=True)
    parser.add_argument("--calib_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--start_s", type=float, default=180.0)
    parser.add_argument("--end_s", type=float, default=200.0)
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--jpeg_quality", type=int, default=92)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    for sub in ["images", "lidar", "ego_pose", "intrinsics", "extrinsics", "sky_masks"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    # --- Calibration ---
    cam_K = {}; cam_dist = {}; cam_size = {}
    for i in range(6):
        K, dist, size = load_intrinsic(
            os.path.join(args.calib_root, "cam_intrinsics", f"cam{i+1}_intrinsic.json"))
        cam_K[i] = K; cam_dist[i] = dist; cam_size[i] = size

    T_lidar_to_cam = {}
    for i in range(6):
        T_lidar_to_cam[i] = load_extrinsic(
            os.path.join(args.calib_root, "lidar_cam_joint_extrinsics", LIDAR_TO_CAM_FILES[i]))

    T_combined_to_imu = load_extrinsic(
        os.path.join(args.calib_root, "lidar_imu_extrinsic.json"))

    # Cam->ego (= IMU = oxts_link), in OpenCV cam convention
    T_cam_to_ego_opencv = {}
    for i in range(6):
        T_cam_to_ego_opencv[i] = T_combined_to_imu @ np.linalg.inv(T_lidar_to_cam[i])

    # Waymo loader applies: cam_to_ego = file @ OPENCV2DATASET
    # We want result T_cam_to_ego_opencv = file @ OPENCV2DATASET
    # => file = T_cam_to_ego_opencv @ inv(OPENCV2DATASET)
    inv_o2d = np.linalg.inv(OPENCV2DATASET)
    extrinsic_for_file = {i: T_cam_to_ego_opencv[i] @ inv_o2d for i in range(6)}

    # --- Write intrinsics + extrinsics ---
    for i in range(6):
        K, dist = cam_K[i], cam_dist[i]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        # cam_dist order from JSON: [k1, k2, p1, p2, k3]
        k1, k2, p1, p2, k3 = dist[:5]
        with open(out_dir / "intrinsics" / f"{i}.txt", "w") as f:
            for v in [fx, fy, cx, cy, k1, k2, p1, p2, k3]:
                f.write(f"{v}\n")
        np.savetxt(out_dir / "extrinsics" / f"{i}.txt", extrinsic_for_file[i])

    # --- Resolve absolute time window ---
    bag_start_ns, bag_end_ns = get_bag_time_range(args.bag_path)
    abs_start = bag_start_ns + int(args.start_s * 1e9)
    abs_end = bag_start_ns + int(args.end_s * 1e9)
    print(f"[info] bag time span: {bag_start_ns} .. {bag_end_ns}  "
          f"(duration {(bag_end_ns - bag_start_ns)/1e9:.1f} s)", flush=True)
    print(f"[info] target window: {abs_start} .. {abs_end}  "
          f"({args.start_s}..{args.end_s} s)", flush=True)

    # --- Pass 1: gather lidar timestamps + tf transforms ---
    # We use mcap log_time_ns throughout because camera image header.stamp is
    # bogus (publisher fills it with a monotonic clock starting from boot).
    # log_time is the bag's record time and is consistent across all topics.
    print("[pass1] scanning lidar + tf...", flush=True)
    lidar_ns = []
    tf_entries = []  # (ns, T_map_to_oxts)
    for m in read_ros2_messages(
        args.bag_path,
        topics=[LIDAR_TOPIC, TF_TOPIC],
        start_time=abs_start,
        end_time=abs_end,
    ):
        topic = m.channel.topic
        ns = m.log_time_ns
        if topic == LIDAR_TOPIC:
            lidar_ns.append(ns)
        elif topic == TF_TOPIC:
            for tr in m.ros_msg.transforms:
                if tr.header.frame_id == "map" and tr.child_frame_id == "oxts_link":
                    tf_entries.append((ns, transform_to_matrix(tr.transform)))
    lidar_ns.sort()
    tf_entries.sort(key=lambda x: x[0])
    print(f"[pass1] lidar msgs in window: {len(lidar_ns)}, tf map->oxts: {len(tf_entries)}",
          flush=True)
    if not lidar_ns:
        sys.exit("ERROR: no front_lidar messages in window. Check start_s/end_s.")
    if not tf_entries:
        sys.exit("ERROR: no /tf map->oxts_link in window.")

    # --- Pick master timestamps at target Hz from lidar timestamps ---
    period_ns = int(1e9 / args.hz)
    master_ns = [lidar_ns[0]]
    for ns in lidar_ns[1:]:
        if ns - master_ns[-1] >= period_ns - period_ns // 10:
            master_ns.append(ns)
    print(f"[pass1] master frames: {len(master_ns)} @ ~{args.hz} Hz", flush=True)

    # --- Compute ego_pose per master frame (nearest tf) ---
    tf_ns_arr = np.array([t[0] for t in tf_entries])
    tf_T_list = [t[1] for t in tf_entries]
    for fi, ns in enumerate(master_ns):
        idx = int(np.argmin(np.abs(tf_ns_arr - ns)))
        delta_ms = abs(int(tf_ns_arr[idx] - ns)) / 1e6
        if delta_ms > 50:
            print(f"  [warn] frame {fi}: nearest tf is {delta_ms:.1f} ms away", flush=True)
        np.savetxt(out_dir / "ego_pose" / f"{fi:03d}.txt", tf_T_list[idx])
    print(f"[pass1] wrote {len(master_ns)} ego_pose files", flush=True)

    master_arr = np.array(master_ns)

    # --- Pass 2: stream images + lidar, buffer best per (frame, topic) ---
    print("[pass2] streaming images + lidar...", flush=True)
    best_img = [[None] * 6 for _ in range(len(master_ns))]  # (delta_ns, msg) per (frame, cam)
    best_lidar = [None] * len(master_ns)

    n_seen = 0
    for m in read_ros2_messages(
        args.bag_path,
        topics=ARENACAM_TOPICS + [LIDAR_TOPIC],
        start_time=abs_start,
        end_time=abs_end,
    ):
        n_seen += 1
        if n_seen % 2000 == 0:
            print(f"  ...seen {n_seen} msgs", flush=True)
        ns = m.log_time_ns
        idx = int(np.argmin(np.abs(master_arr - ns)))
        delta = int(abs(master_arr[idx] - ns))
        topic = m.channel.topic
        if topic == LIDAR_TOPIC:
            cur = best_lidar[idx]
            if cur is None or delta < cur[0]:
                best_lidar[idx] = (delta, m.ros_msg)
        else:
            cam_idx = int(topic[len("/arenacam"):].split("/")[0]) - 1
            cur = best_img[idx][cam_idx]
            if cur is None or delta < cur[0]:
                best_img[idx][cam_idx] = (delta, m.ros_msg)
    print(f"[pass2] total msgs streamed: {n_seen}", flush=True)

    # --- Backfill missing camera frames from nearest available frame in same cam ---
    backfilled = []
    for ci in range(6):
        present = [fi for fi in range(len(master_ns)) if best_img[fi][ci] is not None]
        if not present:
            print(f"  [warn] cam {ci}: NO frames available in window", flush=True)
            continue
        present_arr = np.array(present)
        for fi in range(len(master_ns)):
            if best_img[fi][ci] is None:
                nearest = present[int(np.argmin(np.abs(present_arr - fi)))]
                best_img[fi][ci] = best_img[nearest][ci]
                backfilled.append((fi, ci, nearest))
    if backfilled:
        print(f"  [info] backfilled {len(backfilled)} missing (frame,cam) "
              f"entries from nearest neighbor (first few: {backfilled[:5]})", flush=True)

    # --- Write images + dummy zero sky_masks ---
    print("[write] images + dummy sky_masks...", flush=True)
    for fi in range(len(master_ns)):
        for ci in range(6):
            entry = best_img[fi][ci]
            if entry is None:
                continue
            img = decode_image(entry[1])
            h, w = img.shape[:2]
            cv2.imwrite(str(out_dir / "images" / f"{fi:03d}_{ci}.jpg"),
                        img, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            sky = np.zeros((h, w), dtype=np.uint8)
            cv2.imwrite(str(out_dir / "sky_masks" / f"{fi:03d}_{ci}.png"), sky)

    # --- Write lidar ---
    print("[write] lidar...", flush=True)
    origin_ego = T_combined_to_imu[:3, 3].astype(np.float32)
    for fi in range(len(master_ns)):
        entry = best_lidar[fi]
        if entry is None:
            print(f"  [warn] frame {fi}: no lidar", flush=True)
            continue
        xyz, intensity = decode_pointcloud2(entry[1])
        # Transform points from front_lidar frame (= combined) to ego frame
        pts_ego = transform_points(T_combined_to_imu, xyz.astype(np.float64)).astype(np.float32)
        n = len(pts_ego)
        origins = np.broadcast_to(origin_ego, (n, 3)).copy()
        flows = np.zeros((n, 3), dtype=np.float32)
        flow_class = np.full((n, 1), -1.0, dtype=np.float32)
        ground = np.zeros((n, 1), dtype=np.float32)
        inten = intensity.reshape(-1, 1).astype(np.float32)
        elong = np.zeros((n, 1), dtype=np.float32)
        lid_id = np.zeros((n, 1), dtype=np.float32)  # 0 = front
        out = np.concatenate(
            [origins, pts_ego, flows, flow_class, ground, inten, elong, lid_id],
            axis=1,
        )
        assert out.shape[1] == 14, out.shape
        out.astype(np.float32).tofile(out_dir / "lidar" / f"{fi:03d}.bin")

    # --- Frame info ---
    with open(out_dir / "frame_info.json", "w") as f:
        json.dump({"time_of_day": "Day", "location": "mcity_downtown",
                   "weather": "unknown"}, f)

    print(f"\n[done] wrote {len(master_ns)} frames to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
