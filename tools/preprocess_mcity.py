"""Convert one or more Mcity ROS2 mcap bags into a drivestudio scene.

Multiple bags are stitched into a single scene with frame indices that
continue across bag boundaries. All bags must share the same `map` frame
(true for bags recorded in the same session against the Mcity map).

Mandatory outputs per frame in {out_dir} (the on-disk layout drivestudio's
WaymoPixelSource/WaymoLiDARSource expect — directory structure originally
modeled after the Waymo Open Dataset preprocessing pipeline):
  images/{f:03d}_{c}.jpg         (c in 0..5 = arenacam1..6)
  lidar/{f:03d}.bin              (N x 14 float32, drivestudio lidar schema)
  ego_pose/{f:03d}.txt           (4x4 matrix, oxts_link -> map)
  intrinsics/{c}.txt             (9 scalars: fx, fy, cx, cy, k1, k2, p1, p2, k3)
  extrinsics/{c}.txt             (4x4 matrix, cam -> ego, in FLU cam convention)
  frame_info.json                (stub metadata)

LiDAR points are gathered from /rslidar_{front,left,right}_points and all
transformed via T_combined_to_imu — RoboSense P6 houses all three sub-lidars
at a single coordinate frame (the combined centroid), so the per-lidar
names in the calibration are just labels for which sub-element was used to
calibrate each camera pair. The back lidar is recorded but not used (no
camera covers its FOV). Each point's source is recorded in column 13 of the
14-col bin (0=front, 1=left, 2=right). Camera-to-ego is composed via:
    T_cam_to_ego = T_combined_to_imu @ inv(T_lidar_to_cam)

Usage (single bag):
  python tools/preprocess_mcity.py \
    --bag_path /scratch/.../may8-2026-downtown-p2_0.mcap \
    --calib_root /home/billhong/drivestudio/calibration_files \
    --out_dir /scratch/.../mcity/processed/training/000 \
    --start_s 180 --end_s 200 --hz 10

Usage (stitch multiple bags — repeat --bag_path / --start_s / --end_s in order):
  python tools/preprocess_mcity.py \
    --bag_path /scratch/.../downtown-p2_0.mcap --start_s 145.2 --end_s 165.2 \
    --bag_path /scratch/.../downtown-p3_0.mcap --start_s 0    --end_s 20 \
    --calib_root /home/billhong/drivestudio/calibration_files \
    --out_dir /scratch/.../mcity/processed/training/d2_d3_combined \
    --hz 10
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

# Mcity's oxts_link IMU body frame is FRD (X-fwd, Y-right, Z-down), as
# confirmed by lidar_imu_extrinsic (roll ~180 deg between FLU lidar and IMU).
# drivestudio's loader assumes ego is FLU (X-fwd, Y-left, Z-up) and anchors
# the world to ego_0, so an FRD ego makes the trained splat Z-down (upside
# down in any Z-up viewer like Visor). Redefine ego as FLU by rotating 180
# deg about X:  p_flu = R @ p_frd.
R_FLU_FROM_FRD = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1],
], dtype=np.float64)

ARENACAM_TOPICS = [f"/arenacam{i+1}/images" for i in range(6)]
# Lidar topic -> lidar_id written to column 13 of the 14-col bin. All three
# topics share the same coordinate frame (the P6 combined centroid), so a
# single T_combined_to_imu transforms all of them.
LIDAR_TOPICS = {
    "/rslidar_front_points": 0,
    "/rslidar_left_points":  1,
    "/rslidar_right_points": 2,
}
# Front lidar is the master clock for picking per-frame timestamps.
MASTER_LIDAR_TOPIC = "/rslidar_front_points"
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


def process_bag(bag_path, start_s, end_s, hz, out_dir, frame_offset,
                T_combined_to_imu, jpeg_quality):
    """Process one bag's window and write frames into out_dir starting at
    frame_offset. Returns (frames_written, last_ego_to_map) or
    (0, None) if no frames were written."""
    print(f"\n========== bag: {os.path.basename(bag_path)}  "
          f"window: {start_s}..{end_s} s  frame_offset: {frame_offset} ==========",
          flush=True)

    bag_start_ns, bag_end_ns = get_bag_time_range(bag_path)
    abs_start = bag_start_ns + int(start_s * 1e9)
    abs_end = bag_start_ns + int(end_s * 1e9)
    print(f"[info] bag span: {bag_start_ns} .. {bag_end_ns}  "
          f"(duration {(bag_end_ns - bag_start_ns)/1e9:.1f} s)", flush=True)
    print(f"[info] window:   {abs_start} .. {abs_end}", flush=True)

    # --- Pass 1: gather lidar timestamps + tf transforms ---
    print("[pass1] scanning lidar + tf...", flush=True)
    lidar_ns = []
    tf_entries = []
    for m in read_ros2_messages(
        bag_path, topics=[MASTER_LIDAR_TOPIC, TF_TOPIC],
        start_time=abs_start, end_time=abs_end,
    ):
        topic = m.channel.topic
        ns = m.log_time_ns
        if topic == MASTER_LIDAR_TOPIC:
            lidar_ns.append(ns)
        elif topic == TF_TOPIC:
            for tr in m.ros_msg.transforms:
                if tr.header.frame_id == "map" and tr.child_frame_id == "oxts_link":
                    # Right-multiply by R to redefine source frame from FRD
                    # oxts_link to FLU: ego_flu_to_map = ego_frd_to_map @ R.
                    T = transform_to_matrix(tr.transform) @ R_FLU_FROM_FRD
                    tf_entries.append((ns, T))
    lidar_ns.sort()
    tf_entries.sort(key=lambda x: x[0])
    print(f"[pass1] master-lidar msgs: {len(lidar_ns)}, tf map->oxts: {len(tf_entries)}",
          flush=True)
    if not lidar_ns:
        sys.exit(f"ERROR: no front_lidar in {bag_path} window.")
    if not tf_entries:
        sys.exit(f"ERROR: no /tf map->oxts_link in {bag_path} window.")

    # --- Pick master timestamps at target Hz from lidar timestamps ---
    period_ns = int(1e9 / hz)
    master_ns = [lidar_ns[0]]
    for ns in lidar_ns[1:]:
        if ns - master_ns[-1] >= period_ns - period_ns // 10:
            master_ns.append(ns)
    print(f"[pass1] master frames: {len(master_ns)} @ ~{hz} Hz", flush=True)

    # --- Compute ego_pose per master frame (nearest tf) ---
    tf_ns_arr = np.array([t[0] for t in tf_entries])
    tf_T_list = [t[1] for t in tf_entries]
    ego_to_map_per_frame = []
    for fi_local, ns in enumerate(master_ns):
        idx = int(np.argmin(np.abs(tf_ns_arr - ns)))
        delta_ms = abs(int(tf_ns_arr[idx] - ns)) / 1e6
        if delta_ms > 50:
            print(f"  [warn] local frame {fi_local}: nearest tf is {delta_ms:.1f} ms away",
                  flush=True)
        fi_global = frame_offset + fi_local
        ego_to_map_per_frame.append(tf_T_list[idx])
        np.savetxt(out_dir / "ego_pose" / f"{fi_global:03d}.txt", tf_T_list[idx])
    print(f"[pass1] wrote {len(master_ns)} ego_pose files "
          f"({frame_offset:03d}..{frame_offset + len(master_ns) - 1:03d})",
          flush=True)

    master_arr = np.array(master_ns)

    # --- Pass 2: stream images + lidar, buffer best per (local frame, topic) ---
    print("[pass2] streaming images + lidar...", flush=True)
    best_img = [[None] * 6 for _ in range(len(master_ns))]
    # best_lidar[frame_idx] = {topic: (delta_ns, msg)} per lidar source
    best_lidar = [dict() for _ in range(len(master_ns))]

    n_seen = 0
    for m in read_ros2_messages(
        bag_path, topics=ARENACAM_TOPICS + list(LIDAR_TOPICS.keys()),
        start_time=abs_start, end_time=abs_end,
    ):
        n_seen += 1
        if n_seen % 2000 == 0:
            print(f"  ...seen {n_seen} msgs", flush=True)
        ns = m.log_time_ns
        idx = int(np.argmin(np.abs(master_arr - ns)))
        delta = int(abs(master_arr[idx] - ns))
        topic = m.channel.topic
        if topic in LIDAR_TOPICS:
            cur = best_lidar[idx].get(topic)
            if cur is None or delta < cur[0]:
                best_lidar[idx][topic] = (delta, m.ros_msg)
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
              f"entries (first few: {backfilled[:5]})", flush=True)

    # --- Write images + sky_masks ---
    print("[write] images + sky_masks...", flush=True)
    for fi_local in range(len(master_ns)):
        fi_global = frame_offset + fi_local
        for ci in range(6):
            entry = best_img[fi_local][ci]
            if entry is None:
                continue
            img = decode_image(entry[1])
            h, w = img.shape[:2]
            cv2.imwrite(str(out_dir / "images" / f"{fi_global:03d}_{ci}.jpg"),
                        img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            sky = np.zeros((h, w), dtype=np.uint8)
            cv2.imwrite(str(out_dir / "sky_masks" / f"{fi_global:03d}_{ci}.png"), sky)

    # --- Write lidar (concatenate front + left + right) ---
    print("[write] lidar...", flush=True)
    origin_ego = T_combined_to_imu[:3, 3].astype(np.float32)
    missing_counts = {topic: 0 for topic in LIDAR_TOPICS}
    for fi_local in range(len(master_ns)):
        fi_global = frame_offset + fi_local
        per_topic = best_lidar[fi_local]
        if not per_topic:
            print(f"  [warn] frame {fi_global}: no lidar from any topic", flush=True)
            continue

        parts = []
        for topic, lidar_id in LIDAR_TOPICS.items():
            entry = per_topic.get(topic)
            if entry is None:
                missing_counts[topic] += 1
                continue
            xyz, intensity = decode_pointcloud2(entry[1])
            pts_ego = transform_points(
                T_combined_to_imu, xyz.astype(np.float64)
            ).astype(np.float32)
            n = len(pts_ego)
            origins = np.broadcast_to(origin_ego, (n, 3)).copy()
            flows = np.zeros((n, 3), dtype=np.float32)
            flow_class = np.full((n, 1), -1.0, dtype=np.float32)
            ground = np.zeros((n, 1), dtype=np.float32)
            inten = intensity.reshape(-1, 1).astype(np.float32)
            elong = np.zeros((n, 1), dtype=np.float32)
            lid_id = np.full((n, 1), float(lidar_id), dtype=np.float32)
            parts.append(np.concatenate(
                [origins, pts_ego, flows, flow_class, ground, inten, elong, lid_id],
                axis=1,
            ))

        out = np.concatenate(parts, axis=0)
        assert out.shape[1] == 14, out.shape
        out.astype(np.float32).tofile(out_dir / "lidar" / f"{fi_global:03d}.bin")

    for topic, n_missing in missing_counts.items():
        if n_missing:
            print(f"  [info] {topic}: missing on {n_missing}/{len(master_ns)} frames",
                  flush=True)

    return len(master_ns), ego_to_map_per_frame[0], ego_to_map_per_frame[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag_path", action="append", required=True,
                        help="Bag path. Repeat with matching --start_s/--end_s "
                             "to stitch multiple bags into one scene.")
    parser.add_argument("--start_s", action="append", type=float, required=True,
                        help="Per-bag start time (s). Repeat once per --bag_path.")
    parser.add_argument("--end_s", action="append", type=float, required=True,
                        help="Per-bag end time (s). Repeat once per --bag_path.")
    parser.add_argument("--calib_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--jpeg_quality", type=int, default=92)
    args = parser.parse_args()

    if not (len(args.bag_path) == len(args.start_s) == len(args.end_s)):
        sys.exit(f"ERROR: --bag_path/--start_s/--end_s counts mismatch: "
                 f"{len(args.bag_path)}/{len(args.start_s)}/{len(args.end_s)}")

    # Infor print on user input for what bags to stitch together
    n_bags = len(args.bag_path)
    print(f"[stitch] {n_bags} bag(s) -> {args.out_dir}", flush=True)
    for i, (bp, s_s, e_s) in enumerate(zip(args.bag_path, args.start_s, args.end_s)):
        print(f"  [{i}] {bp}  start={s_s}s  end={e_s}s  dur={e_s - s_s:.2f}s",
              flush=True)

    out_dir = Path(args.out_dir)
    for sub in ["images", "lidar", "ego_pose", "intrinsics", "extrinsics", "sky_masks"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    # --- Calibration (shared across all bags) ---
    cam_K = {}; cam_dist = {}; cam_size = {}
    for i in range(6):
        K, dist, size = load_intrinsic(
            os.path.join(args.calib_root, "cam_intrinsics", f"cam{i+1}_intrinsic.json"))
        cam_K[i] = K; cam_dist[i] = dist; cam_size[i] = size

    T_lidar_to_cam = {}
    for i in range(6):
        T_lidar_to_cam[i] = load_extrinsic(
            os.path.join(args.calib_root, "lidar_cam_joint_extrinsics", LIDAR_TO_CAM_FILES[i]))

    T_combined_to_imu_frd = load_extrinsic(
        os.path.join(args.calib_root, "lidar_imu_extrinsic.json"))
    # Re-express in FLU ego frame so downstream artifacts match drivestudio's
    # FLU/Z-up convention.
    T_combined_to_imu = R_FLU_FROM_FRD @ T_combined_to_imu_frd

    # Cam->ego (= IMU = oxts_link, redefined as FLU), in OpenCV cam convention
    T_cam_to_ego_opencv = {}
    for i in range(6):
        T_cam_to_ego_opencv[i] = T_combined_to_imu @ np.linalg.inv(T_lidar_to_cam[i])

    # drivestudio's loader applies: cam_to_ego = file @ OPENCV2DATASET
    # => file = T_cam_to_ego_opencv @ inv(OPENCV2DATASET)
    inv_o2d = np.linalg.inv(OPENCV2DATASET)
    extrinsic_for_file = {i: T_cam_to_ego_opencv[i] @ inv_o2d for i in range(6)}

    # --- Write intrinsics + extrinsics (once) ---
    for i in range(6):
        K, dist = cam_K[i], cam_dist[i]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        k1, k2, p1, p2, k3 = dist[:5]
        with open(out_dir / "intrinsics" / f"{i}.txt", "w") as f:
            for v in [fx, fy, cx, cy, k1, k2, p1, p2, k3]:
                f.write(f"{v}\n")
        np.savetxt(out_dir / "extrinsics" / f"{i}.txt", extrinsic_for_file[i])

    # --- Process each bag, accumulating frame indices ---
    frame_offset = 0
    prev_last_ego_to_map = None
    segments_meta = []
    for bag_path, s_s, e_s in zip(args.bag_path, args.start_s, args.end_s):
        n, first_ego, last_ego = process_bag(
            bag_path, s_s, e_s, args.hz, out_dir, frame_offset,
            T_combined_to_imu, args.jpeg_quality)
        segments_meta.append({
            "bag": os.path.basename(bag_path),
            "start_s": s_s, "end_s": e_s,
            "frame_start": frame_offset, "frame_end": frame_offset + n - 1,
            "num_frames": n,
        })
        # Sanity check: distance between previous bag's last ego_to_map
        # translation and this bag's first. Large jump => map frames disagree.
        if prev_last_ego_to_map is not None:
            d = np.linalg.norm(first_ego[:3, 3] - prev_last_ego_to_map[:3, 3])
            print(f"[stitch] gap to previous bag's last ego pose: {d:.2f} m "
                  f"(large >> ~vehicle-motion-between-bags suggests map-frame mismatch)",
                  flush=True)
        prev_last_ego_to_map = last_ego
        frame_offset += n

    # --- Frame info ---
    with open(out_dir / "frame_info.json", "w") as f:
        json.dump({
            "time_of_day": "Day",
            "location": "mcity_downtown",
            "weather": "unknown",
            "segments": segments_meta,
        }, f, indent=2)

    print(f"\n[done] wrote {frame_offset} frames to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
