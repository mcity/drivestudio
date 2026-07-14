"""Convert a Mcity ROS2 recording (split across simultaneous mcap bags) into a
drivestudio scene.

A session is recorded as separate, simultaneous bags that all span the same
time: camera bag(s) with /arenacam*/images (e.g. camA = cams 1-3, camB = cams
4-6) and one lidar bag with /rslidar_points + /tf + /ins/*. Sensors are
PTP-synced, so frames are paired by sensor **header** timestamp, NOT bag/log
time. The cameras stamp headers in TAI, +37 s ahead of the lidar/INS/tf UTC
clock (the TAI-UTC leap offset), so --cam_offset_s (default 37) is subtracted
from every camera header to bring it onto the lidar/tf reference timeline.

Mandatory outputs per frame in {out_dir} (the on-disk layout drivestudio's
WaymoPixelSource/WaymoLiDARSource expect — directory structure originally
modeled after the Waymo Open Dataset preprocessing pipeline):
  images/{f:03d}_{c}.jpg         (c in 0..5 = arenacam1..6)
  lidar/{f:03d}.bin              (N x 14 float32, drivestudio lidar schema)
  ego_pose/{f:03d}.txt           (4x4 matrix, ego(FLU) -> map)
  intrinsics/{c}.txt             (9 scalars: fx, fy, cx, cy, k1, k2, p1, p2, k3)
  extrinsics/{c}.txt             (4x4 matrix, cam -> ego, in FLU cam convention)
  frame_info.json                (stub metadata)

LiDAR points come from the single RoboSense Ruby+ lidar (/rslidar_points) and
are transformed into the ego frame via T_lidar_to_ego. The ego is the
GPS/RTK-fixed IMU (oxts_link), redefined FLU (X-fwd/Y-left/Z-up) so drivestudio
renders Z-up. Column 13 of the 14-col bin records the lidar source id (always 0
— single lidar). Camera-to-ego composes straight through the lidar, since the
cam<->lidar calibration is already T_cam_to_lidar (OpenCV optical convention):
    T_imu_to_lidar = [ R(q_ItoL) | p_IinL ]          (from lidar<->IMU calib)
    T_lidar_to_ego = R_FLU_FROM_FRD @ inv(T_imu_to_lidar)
    T_cam_to_ego   = T_lidar_to_ego @ T_cam_to_lidar

Calibration files under {calib_root} (MATLAB / OA-LICalib formats):
  cam_intrinsics/intrinsics_matlab{1..6}.json            (K, dist, image size)
  lidar_cam_extrinsics/cam_to_lidar_matrices_matlab_cam{1..6}.json  (T_cam_to_lidar)
  lidar_imu_extrinsic.json                               (p_IinL, q_ItoL)

Usage:
  python tools/preprocess_mcity.py \
    --lidar_bag /scratch/.../july2-2026/zone_2/lidar_ins_tf/lidar_ins_tf_0.mcap \
    --cam_bag   /scratch/.../july2-2026/zone_2/camA/camA_0.mcap \
    --cam_bag   /scratch/.../july2-2026/zone_2/camB/camB_0.mcap \
    --calib_root /home/billhong/drivestudio/calibration_files \
    --out_dir /scratch/.../mcity/processed/training/zone_2 \
    --start_s 0 --end_s 30 --hz 10

--start_s/--end_s are relative to the first /rslidar_points header stamp;
use --end_s end for the whole recording.
"""
import argparse
import json
import os
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory


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
# Single RoboSense Ruby+ lidar; its points get source id 0 in column 13 of the
# 14-col bin. Also the master clock for choosing per-frame timestamps.
MASTER_LIDAR_TOPIC = "/rslidar_points"
TF_TOPIC = "/tf"


def load_intrinsic(path):
    """MATLAB-format intrinsics: camera_matrix (3x3), dist_coeffs
    (k1,k2,p1,p2,k3), image_width/height."""
    with open(path) as f:
        d = json.load(f)
    K = np.array(d["camera_matrix"], dtype=np.float64)
    dist = np.array(d["dist_coeffs"], dtype=np.float64).flatten()
    w, h = d["image_width"], d["image_height"]
    return K, dist, (w, h)


def load_cam_to_lidar(path):
    """MATLAB-format cam<->lidar extrinsic: a single 'cam_<n>' key holding a 4x4
    T_cam_to_lidar in OpenCV optical convention (p_lidar = T @ p_cam,
    camera x-right / y-down / z-forward)."""
    with open(path) as f:
        d = json.load(f)
    key = next(k for k in d if k.startswith("cam_"))
    return np.array(d[key], dtype=np.float64)


def load_lidar_imu(path):
    """Lidar<->IMU calib -> T_imu_to_lidar (4x4). p_IinL is the IMU origin
    expressed in the lidar frame; q_ItoL is the IMU->lidar rotation (x,y,z,w),
    so a point's IMU coords map to lidar coords via p_L = R(q_ItoL) @ p_I + p_IinL."""
    with open(path) as f:
        d = json.load(f)
    p = d["lidar_imu_extrinsic"]["param"]
    t, q = p["p_IinL"], p["q_ItoL"]
    T = np.eye(4)
    T[:3, :3] = quat_to_R(q["x"], q["y"], q["z"], q["w"])
    T[:3, 3] = [t["x"], t["y"], t["z"]]
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


def hdr_stamp_ns(data):
    """std_msgs/Header stamp (ns) straight from raw CDR bytes — no decode. Image
    and PointCloud2 both begin with std_msgs/Header, so after the 4-byte CDR
    encapsulation the layout is int32 sec, uint32 nanosec (the same trick the
    LCECalib exporter uses). Avoids decoding 230k-point clouds just to time them."""
    sec, nsec = struct.unpack_from("<iI", data, 4)
    return sec * 1_000_000_000 + nsec


def first_lidar_header_ns(lidar_bag):
    """UTC header stamp (ns) of the first /rslidar_points scan — the scene time
    origin that --start_s/--end_s are measured from."""
    with open(lidar_bag, "rb") as f:
        for _, ch, msg in make_reader(f).iter_messages(topics=[MASTER_LIDAR_TOPIC]):
            return hdr_stamp_ns(msg.data)
    sys.exit(f"ERROR: no {MASTER_LIDAR_TOPIC} in {lidar_bag}")


def process_scene(cam_bags, lidar_bag, start_s, end_s, hz, out_dir,
                  T_lidar_to_ego, jpeg_quality, cam_offset_s):
    """Convert one recording session (simultaneous split bags) into frames
    [0..N). Pairing uses PTP sensor header stamps on the lidar/INS/tf UTC clock;
    camera header stamps are shifted by -cam_offset_s (TAI->UTC) onto it.
    Returns (num_frames, end_s_resolved)."""
    print(f"\n========== scene: lidar={os.path.basename(lidar_bag)}  "
          f"cams={[os.path.basename(b) for b in cam_bags]} ==========", flush=True)
    cam_off_ns = int(round(cam_offset_s * 1e9))
    # Log-time slack for the windowed reads: every bag's log_time sits ~UTC+0..0.3s,
    # so [win-MARGIN, win+MARGIN] on log_time over-captures the UTC-header window
    # (incl. cameras, whose log ~ UTC despite their +37s TAI header stamps).
    MARGIN_NS = 2_000_000_000

    # --- Window on the lidar UTC-header timeline (start_s/end_s from 1st scan) ---
    u0 = first_lidar_header_ns(lidar_bag)
    _, lidar_log_end = get_bag_time_range(lidar_bag)
    win_start = u0 + int(start_s * 1e9)
    if isinstance(end_s, str) and end_s == "end":
        win_end = None
        end_disp = (lidar_log_end - u0) / 1e9
    else:
        win_end = u0 + int(end_s * 1e9)
        end_disp = float(end_s)
    log_lo = win_start - MARGIN_NS
    log_hi = lidar_log_end if win_end is None else win_end + MARGIN_NS

    def in_win(u):
        return u >= win_start and (win_end is None or u <= win_end)

    print(f"[info] window {start_s}..{end_disp:.2f}s on lidar UTC header clock; "
          f"cam_offset={cam_offset_s}s", flush=True)

    # --- Pass 1 (lidar bag): master lidar header stamps + tf map->oxts_link ---
    print("[pass1] scanning lidar headers + tf...", flush=True)
    lidar_ns = []
    tf_entries = []
    with open(lidar_bag, "rb") as f:
        fac = DecoderFactory()
        for schema, ch, msg in make_reader(f).iter_messages(
                topics=[MASTER_LIDAR_TOPIC, TF_TOPIC],
                start_time=log_lo, end_time=log_hi):
            if ch.topic == MASTER_LIDAR_TOPIC:
                u = hdr_stamp_ns(msg.data)
                if in_win(u):
                    lidar_ns.append(u)
            else:
                for tr in fac.decoder_for("cdr", schema)(msg.data).transforms:
                    if tr.header.frame_id == "map" and tr.child_frame_id == "oxts_link":
                        u = stamp_to_ns(tr.header.stamp)
                        if in_win(u):
                            # ego_flu_to_map = ego_frd_to_map @ R_FLU_FROM_FRD
                            T = transform_to_matrix(tr.transform) @ R_FLU_FROM_FRD
                            tf_entries.append((u, T))
    lidar_ns.sort()
    tf_entries.sort(key=lambda x: x[0])
    print(f"[pass1] master-lidar msgs: {len(lidar_ns)}, tf map->oxts: {len(tf_entries)}",
          flush=True)
    if not lidar_ns:
        sys.exit(f"ERROR: no {MASTER_LIDAR_TOPIC} in window.")
    if not tf_entries:
        sys.exit("ERROR: no /tf map->oxts_link in window.")

    # --- Subsample master timestamps to target Hz ---
    period_ns = int(1e9 / hz)
    master_ns = [lidar_ns[0]]
    for u in lidar_ns[1:]:
        if u - master_ns[-1] >= period_ns - period_ns // 10:
            master_ns.append(u)
    N = len(master_ns)
    master_arr = np.array(master_ns)
    print(f"[pass1] master frames: {N} @ ~{hz} Hz", flush=True)

    # --- ego_pose per master frame (nearest tf by header stamp) ---
    tf_ns_arr = np.array([t[0] for t in tf_entries])
    tf_T_list = [t[1] for t in tf_entries]
    for i, u in enumerate(master_ns):
        j = int(np.argmin(np.abs(tf_ns_arr - u)))
        dms = abs(int(tf_ns_arr[j] - u)) / 1e6
        if dms > 50:
            print(f"  [warn] frame {i}: nearest tf is {dms:.1f} ms away", flush=True)
        np.savetxt(out_dir / "ego_pose" / f"{i:03d}.txt", tf_T_list[j])
    print(f"[pass1] wrote {N} ego_pose files", flush=True)

    # --- Pass 2a (lidar bag): keep the scan nearest each master frame ---
    print("[pass2a] streaming lidar clouds...", flush=True)
    best_lidar = [None] * N   # (delta_ns, decoded_msg)
    with open(lidar_bag, "rb") as f:
        fac = DecoderFactory()
        for schema, ch, msg in make_reader(f).iter_messages(
                topics=[MASTER_LIDAR_TOPIC], start_time=log_lo, end_time=log_hi):
            u = hdr_stamp_ns(msg.data)
            if not in_win(u):
                continue
            i = int(np.argmin(np.abs(master_arr - u)))
            d = abs(int(master_arr[i] - u))
            if best_lidar[i] is None or d < best_lidar[i][0]:
                best_lidar[i] = (d, fac.decoder_for("cdr", schema)(msg.data))

    # --- Pass 2b (camera bags): keep the frame nearest each master, per cam.
    #     Camera header (TAI) is shifted by -cam_off_ns onto the lidar UTC clock. ---
    print("[pass2b] streaming camera images...", flush=True)
    best_img = [[None] * 6 for _ in range(N)]
    for cb in cam_bags:
        n_win = 0
        with open(cb, "rb") as f:
            fac = DecoderFactory()   # fresh per file: mcap schema ids are per-file
            for schema, ch, msg in make_reader(f).iter_messages(
                    topics=ARENACAM_TOPICS, start_time=log_lo, end_time=log_hi):
                u = hdr_stamp_ns(msg.data) - cam_off_ns
                if not in_win(u):
                    continue
                n_win += 1
                i = int(np.argmin(np.abs(master_arr - u)))
                d = abs(int(master_arr[i] - u))
                ci = int(ch.topic[len("/arenacam"):].split("/")[0]) - 1
                if best_img[i][ci] is None or d < best_img[i][ci][0]:
                    best_img[i][ci] = (d, fac.decoder_for("cdr", schema)(msg.data))
        print(f"  {os.path.basename(cb)}: {n_win} in-window camera msgs", flush=True)

    # --- Backfill missing (frame,cam) from the nearest available frame per cam ---
    backfilled = 0
    for ci in range(6):
        present = [i for i in range(N) if best_img[i][ci] is not None]
        if not present:
            print(f"  [warn] cam {ci} (arenacam{ci+1}): NO frames in window", flush=True)
            continue
        present_arr = np.array(present)
        for i in range(N):
            if best_img[i][ci] is None:
                nn = present[int(np.argmin(np.abs(present_arr - i)))]
                best_img[i][ci] = best_img[nn][ci]
                backfilled += 1
    if backfilled:
        print(f"  [info] backfilled {backfilled} missing (frame,cam) entries", flush=True)

    # --- Write images + sky_masks ---
    print("[write] images + sky_masks...", flush=True)
    for i in range(N):
        for ci in range(6):
            entry = best_img[i][ci]
            if entry is None:
                continue
            img = decode_image(entry[1])
            h, w = img.shape[:2]
            cv2.imwrite(str(out_dir / "images" / f"{i:03d}_{ci}.jpg"),
                        img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            cv2.imwrite(str(out_dir / "sky_masks" / f"{i:03d}_{ci}.png"),
                        np.zeros((h, w), dtype=np.uint8))

    # --- Write lidar (single lidar -> source id 0 in column 13) ---
    print("[write] lidar...", flush=True)
    origin_ego = T_lidar_to_ego[:3, 3].astype(np.float32)
    n_missing = 0
    for i in range(N):
        if best_lidar[i] is None:
            n_missing += 1
            print(f"  [warn] frame {i}: no lidar in window", flush=True)
            continue
        xyz, intensity = decode_pointcloud2(best_lidar[i][1])
        pts_ego = transform_points(
            T_lidar_to_ego, xyz.astype(np.float64)).astype(np.float32)
        n = len(pts_ego)
        origins = np.broadcast_to(origin_ego, (n, 3)).copy()
        flows = np.zeros((n, 3), dtype=np.float32)
        flow_class = np.full((n, 1), -1.0, dtype=np.float32)
        ground = np.zeros((n, 1), dtype=np.float32)
        inten = intensity.reshape(-1, 1).astype(np.float32)
        elong = np.zeros((n, 1), dtype=np.float32)
        lid_id = np.zeros((n, 1), dtype=np.float32)   # single lidar -> id 0
        out = np.concatenate(
            [origins, pts_ego, flows, flow_class, ground, inten, elong, lid_id], axis=1)
        assert out.shape[1] == 14, out.shape
        out.astype(np.float32).tofile(out_dir / "lidar" / f"{i:03d}.bin")
    if n_missing:
        print(f"  [info] lidar missing on {n_missing}/{N} frames", flush=True)

    return N, end_disp


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lidar_bag", required=True,
                        help="bag with /rslidar_points + /tf + /ins/* (e.g. lidar_ins_tf).")
    parser.add_argument("--cam_bag", action="append", required=True,
                        help="camera bag with /arenacam*/images; repeat once per bag "
                             "(e.g. camA for cams 1-3, camB for cams 4-6).")
    parser.add_argument("--start_s", type=float, default=0.0,
                        help="window start (s) from the first /rslidar_points header.")
    parser.add_argument("--end_s", type=lambda v: v if v == "end" else float(v),
                        default="end", help="window end (s), or 'end' for the whole bag.")
    parser.add_argument("--cam_offset_s", type=float, default=37.0,
                        help="seconds subtracted from camera header stamps to align them "
                             "(TAI) to the lidar/INS/tf clock (UTC). TAI-UTC = 37 s.")
    parser.add_argument("--calib_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--jpeg_quality", type=int, default=92)
    args = parser.parse_args()

    print(f"[scene] lidar_bag = {args.lidar_bag}", flush=True)
    for b in args.cam_bag:
        print(f"[scene] cam_bag   = {b}", flush=True)
    print(f"[scene] -> {args.out_dir}", flush=True)

    out_dir = Path(args.out_dir)
    for sub in ["images", "lidar", "ego_pose", "intrinsics", "extrinsics", "sky_masks"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    # --- Calibration ---
    cam_K = {}; cam_dist = {}; cam_size = {}
    for i in range(6):
        K, dist, size = load_intrinsic(
            os.path.join(args.calib_root, "cam_intrinsics", f"intrinsics_matlab{i+1}.json"))
        cam_K[i] = K; cam_dist[i] = dist; cam_size[i] = size

    # T_cam_to_lidar[i]: camera i (OpenCV optical) -> lidar frame, read directly.
    T_cam_to_lidar = {}
    for i in range(6):
        T_cam_to_lidar[i] = load_cam_to_lidar(os.path.join(
            args.calib_root, "lidar_cam_extrinsics",
            f"cam_to_lidar_matrices_matlab_cam{i+1}.json"))

    # Lidar -> ego. Ego is the GPS/RTK-fixed IMU (oxts_link, FRD), redefined FLU
    # (X-fwd/Y-left/Z-up) so drivestudio renders Z-up; ego_pose applies the same
    # R_FLU_FROM_FRD. T_imu_to_lidar comes from the lidar<->IMU calibration.
    T_imu_to_lidar = load_lidar_imu(
        os.path.join(args.calib_root, "lidar_imu_extrinsic.json"))
    T_lidar_to_ego = R_FLU_FROM_FRD @ np.linalg.inv(T_imu_to_lidar)

    # Cam->ego (OpenCV cam convention), composed straight through the lidar.
    T_cam_to_ego_opencv = {
        i: T_lidar_to_ego @ T_cam_to_lidar[i] for i in range(6)
    }

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

    # --- Process the scene (one session across the simultaneous bags) ---
    n, end_disp = process_scene(
        args.cam_bag, args.lidar_bag, args.start_s, args.end_s, args.hz,
        out_dir, T_lidar_to_ego, args.jpeg_quality, args.cam_offset_s)

    # --- Frame info ---
    with open(out_dir / "frame_info.json", "w") as f:
        json.dump({
            "time_of_day": "Day",
            "location": "mcity",
            "weather": "unknown",
            "num_frames": n,
            "start_s": args.start_s, "end_s": end_disp, "hz": args.hz,
            "cam_offset_s": args.cam_offset_s,
            "lidar_bag": os.path.basename(args.lidar_bag),
            "cam_bags": [os.path.basename(b) for b in args.cam_bag],
        }, f, indent=2)

    print(f"\n[done] wrote {n} frames to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
