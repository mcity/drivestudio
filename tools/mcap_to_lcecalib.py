#!/usr/bin/env python3
"""Select specific timestamps from a RoboSense-128 + Arena-camera ROS2 .mcap and
export LiDAR/-camera pairs in the folder layout LCECalib expects.

LCECalib (run_lcecalib_fe.m) reads a dataset laid out as:
    <data_path>/img/000000.png , 000001.png , ...   (raw distorted images; +params.mat)
    <data_path>/pcd/000000.pcd , 000001.pcd , ...   (binary PCD: x y z intensity,
                                                     where 'intensity' holds RING)
index i of img/ is paired with index i of pcd/ (same 6-digit stem).

This script:
  * LiDAR  /rslidar_points  (organized 1800x128, x/y/z/intensity f32, ring u16,
    timestamp f64; is_dense=False)  ->  binary .pcd with FIELDS x y z intensity,
    invalid (NaN) points dropped. MATLAB pcread() populates Location + Intensity.
    NB: the 'intensity' field carries the LiDAR RING (beam index 0..127), because
    LCECalib's board extraction reads pc.Intensity as the ring (--pcd-ch4).
  * Camera /arenacamN/images (bayer_rggb8, 1920x1200) -> debayered BGR .png
    (raw/distorted, as LCECalib undistorts internally with K,D from params.mat).

For every requested timestamp it picks the nearest LiDAR scan (by sensor header
stamp, NOT bag record time) and then the camera frame nearest that chosen scan,
and reports the LiDAR<->camera offset so you can judge sync quality.

NB: in this bag the camera stamps its header on a different clock, leading the
LiDAR/recorder clock by ~37s (constant). That offset is auto-estimated and
subtracted before pairing; override with --cam-offset. Residual sync accuracy is
a few hundred ms at worst (differing record latencies), fine for a static board.

Modes
-----
  list     : print every LiDAR frame's relative time (pick timestamps from this).
  preview  : dump downsized JPEGs of the camera stream every --every seconds,
             named by relative time, so you can eyeball where the board is.
  extract  : write img/000000.png + pcd/000000.pcd ... for the chosen timestamps.

Runs on the `drivestudio` conda env (has mcap, mcap-ros2-support, numpy, opencv):
    /home/billhong/.conda/envs/drivestudio/bin/python mcap_to_lcecalib.py ...

Examples
--------
  # 1) see what's in the bag
  ... mcap_to_lcecalib.py list

  # 2) dump a preview JPEG every 1s to browse for good (static board) frames
  ... mcap_to_lcecalib.py preview --every 1.0

  # 3a) extract specific relative-time frames (seconds from bag start)
  ... mcap_to_lcecalib.py extract --times 5.0,12.4,23.1,40.0

  # 3b) or sample uniformly, checking the pairing first
  ... mcap_to_lcecalib.py extract --start 5 --end 78 --every 3 --dry-run
  ... mcap_to_lcecalib.py extract --start 5 --end 78 --every 3

  # 3c) two bags, separate timestamps each, pooled into ONE dir with continuous
  #     numbering (camera inferred as arenacam2 from 'lidar-cam2'):
  ... mcap_to_lcecalib.py extract lidar-cam2_0.mcap --mcap2 lidar-cam2_1.mcap \
        --times 1.8,7.3,15.2 --times2 3.0,9.5,20.1 --outdir .../arenacam2
"""
import argparse
import os
import re
import struct
import sys

import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

DEFAULT_MCAP = "/scratch/mcity_project_root/mcity_project/billhong/LCECalib/lidar-cam1_0.mcap"
LIDAR_TOPIC = "/rslidar_points"

# sensor_msgs/PointField datatype code -> numpy format
ROS_NP = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}

# ROS bayer encoding -> OpenCV conversion code (cv_bridge convention: names shifted)
BAYER2CV = {
    "bayer_rggb8": "COLOR_BayerBG2BGR",
    "bayer_bggr8": "COLOR_BayerRG2BGR",
    "bayer_gbrg8": "COLOR_BayerGR2BGR",
    "bayer_grbg8": "COLOR_BayerGB2BGR",
}


def cam_topic(name):
    """Accept 'arenacam1', '1', or a full '/arenacam1/images'."""
    if name.startswith("/"):
        return name
    if name.isdigit():
        name = "arenacam" + name
    return f"/{name}/images"


def cam_tag(topic):
    return topic.strip("/").split("/")[0]  # /arenacam1/images -> arenacam1


# ----------------------------------------------------------------------------- decode
def pc2_to_xyz4(msg, ch4="ring"):
    """PointCloud2 -> (N,4) float32 [x, y, z, ch4], finite xyz only.

    ch4 goes into the PCD 'intensity' field, which is the only extra scalar
    MATLAB pcread() surfaces. LCECalib's board extraction (boardpts_ext) needs the
    LiDAR RING (beam index) in that column to reconstruct scan lines, so ch4
    defaults to 'ring' (0..127), NOT reflectivity. Use ch4='intensity' if you want
    true reflectivity instead (board extraction will then fail — LCECalib-specific)."""
    names, formats, offsets = [], [], []
    present = set()
    for f in msg.fields:
        if f.count != 1:
            continue
        names.append(f.name)
        formats.append(ROS_NP[f.datatype])
        offsets.append(f.offset)
        present.add(f.name)
    dt = np.dtype({"names": names, "formats": formats,
                   "offsets": offsets, "itemsize": msg.point_step})
    a = np.frombuffer(msg.data, dtype=dt)
    x = a["x"].astype(np.float32)
    y = a["y"].astype(np.float32)
    z = a["z"].astype(np.float32)
    if ch4 not in present:
        raise ValueError(f"field {ch4!r} not in cloud (have {sorted(present)})")
    c = a[ch4].astype(np.float32)
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    return np.stack([x[good], y[good], z[good], c[good]], axis=1)


def write_pcd(path, xyzi):
    n = len(xyzi)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(np.ascontiguousarray(xyzi, dtype="<f4").tobytes())


def image_to_bgr(msg):
    import cv2
    enc = msg.encoding
    buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    if enc in BAYER2CV:
        out = cv2.cvtColor(buf[:, : msg.width], getattr(cv2, BAYER2CV[enc]))
    elif enc == "rgb8":
        out = cv2.cvtColor(buf[:, : msg.width * 3].reshape(msg.height, msg.width, 3),
                           cv2.COLOR_RGB2BGR)
    elif enc == "bgr8":
        out = buf[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
    elif enc == "mono8":
        out = buf[:, : msg.width]
    else:
        raise ValueError(f"unsupported image encoding: {enc!r}")
    # reverse x & y: 180-degree rotation (mirror horizontally + vertically)
    return cv2.rotate(out, cv2.ROTATE_180)


def stamp_ns(data):
    """Header.stamp (ns) straight from raw CDR bytes — no decode. Every msg here
    (Image/PointCloud2) starts with std_msgs/Header, so after the 4-byte CDR
    encapsulation the layout is int32 sec, uint32 nanosec."""
    sec, nsec = struct.unpack_from("<iI", data, 4)
    return sec * 1_000_000_000 + nsec


# ----------------------------------------------------------------------------- passes
def bag_start(path):
    """Global bag start (ns) from the mcap summary — no full scan."""
    with open(path, "rb") as f:
        s = make_reader(f).get_summary()
    return s.statistics.message_start_time


def index_topics(path, topics):
    """One pass (no CDR decode): sorted per-topic header-stamp (sensor time) lists.
    Uses the message header, NOT log_time — bag record time is bursty/buffered."""
    idx = {t: [] for t in topics}
    with open(path, "rb") as f:
        for _, channel, message in make_reader(f).iter_messages(topics=topics):
            idx[channel.topic].append(stamp_ns(message.data))
    for t in topics:
        idx[t].sort()
    return idx


def estimate_cam_offset(path, cam, lidar=LIDAR_TOPIC, need=60):
    """Estimate the camera->LiDAR header-clock offset (seconds).

    The two sensors stamp headers in different clocks; the shared reference is the
    recorder's log_time. Per topic, (header - log) = clock_offset - record_latency,
    so median(header-log)_cam - median(header-log)_lid cancels the (assumed-similar)
    latencies and yields how far the camera clock leads the LiDAR clock. Reads only
    until `need` samples of each topic are seen (start of file), so it is cheap.

    NOTE: only good to a few hundred ms (differing record latencies); override with
    --cam-offset if you have better sync knowledge."""
    import statistics
    dl, dc = [], []
    with open(path, "rb") as f:
        for _, ch, m in make_reader(f).iter_messages(topics=[lidar, cam]):
            d = stamp_ns(m.data) - m.log_time
            if ch.topic == lidar and len(dl) < need:
                dl.append(d)
            elif ch.topic == cam and len(dc) < need:
                dc.append(d)
            if len(dl) >= need and len(dc) >= need:
                break
    if not dc:
        return 0.0
    off = statistics.median(dc) - (statistics.median(dl) if dl else 0)
    return off / 1e9


def mode_list(args):
    start = bag_start(args.mcap)
    idx = index_topics(args.mcap, [LIDAR_TOPIC, args.cam])
    lid = idx[LIDAR_TOPIC]
    cam = idx[args.cam]
    cam_off = args.cam_offset if args.cam_offset is not None \
        else estimate_cam_offset(args.mcap, args.cam)
    cam_off_ns = int(round(cam_off * 1e9))
    print(f"bag_start_ns = {start}")
    print(f"camera->lidar clock offset = {cam_off:.3f}s (applied to camera times below)")
    for t, arr, off in ((LIDAR_TOPIC, lid, 0), (args.cam, cam, cam_off_ns)):
        if arr:
            print(f"{t:24s} n={len(arr):5d}  "
                  f"rel=[{(arr[0]-off-start)/1e9:.2f}, {(arr[-1]-off-start)/1e9:.2f}]s")
    print(f"\n{'idx':>4s} {'rel_time_s':>11s}  (LiDAR frames — pick these for --times)")
    for i, lt in enumerate(lid):
        print(f"{i:>4d} {(lt-start)/1e9:>11.3f}")


def mode_preview(args):
    import cv2
    start = bag_start(args.mcap)
    cam_off = args.cam_offset if args.cam_offset is not None \
        else estimate_cam_offset(args.mcap, args.cam)
    cam_off_ns = int(round(cam_off * 1e9))
    outdir = args.preview_dir or os.path.join(
        os.path.dirname(args.mcap), f"preview_{cam_tag(args.cam)}")
    os.makedirs(outdir, exist_ok=True)
    step_ns = int(args.every * 1e9)
    fac = DecoderFactory()
    next_t = start
    n = 0
    with open(args.mcap, "rb") as f:
        for schema, channel, message in make_reader(f).iter_messages(topics=[args.cam]):
            s = stamp_ns(message.data) - cam_off_ns  # aligned to LiDAR clock
            if s < next_t:
                continue
            msg = fac.decoder_for("cdr", schema)(message.data)
            bgr = image_to_bgr(msg)
            if args.scale != 1.0:
                bgr = cv2.resize(bgr, None, fx=args.scale, fy=args.scale)
            rel = (s - start) / 1e9
            cv2.imwrite(os.path.join(outdir, f"t{rel:07.2f}s.jpg"), bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            n += 1
            next_t += step_ns
    print(f"wrote {n} preview JPEGs (every {args.every}s) -> {outdir}")


def parse_targets(args, start, times=None, times_file=None):
    """Return sorted list of absolute target sensor-times in ns for one bag.
    `times`/`times_file` override args.times/args.times_file (per-bag lists);
    the uniform --start/--end/--every fallback applies only when no list is given."""
    times = times if times is not None else args.times
    times_file = times_file if times_file is not None else args.times_file
    if times or times_file:
        raw = times or ""
        if times_file:
            with open(times_file) as f:
                raw += " " + f.read()
        vals = [float(x) for x in raw.replace(",", " ").split()]
        if not vals:
            sys.exit("no timestamps parsed from --times/--times-file")
        return sorted(int(v * 1e9) if args.abs else start + int(v * 1e9) for v in vals)
    if args.start is not None and args.end is not None and args.every:
        rels = np.arange(args.start, args.end + 1e-9, args.every)
        return sorted(start + int(r * 1e9) for r in rels)
    sys.exit("choose timestamps: --times / --times-file, or --start --end --every")


def _extract_bag_rows(args, mcap, times, times_file, label, decode):
    """Stream ONE bag and return its paired rows.
    Each bag is timed independently (targets relative to that bag's start, camera
    offset estimated per bag). When `decode` is set, the chosen messages are
    decoded HERE with a fresh per-bag DecoderFactory and the final PCD array + BGR
    image are stored in the row -- mcap schema ids are per-file, so decoding both
    bags through one shared factory would collide ids and mis-decode the 2nd bag."""
    topics = [LIDAR_TOPIC, args.cam]
    start = bag_start(mcap)
    targets = parse_targets(args, start, times, times_file)
    N = len(targets)
    cam_off = args.cam_offset if args.cam_offset is not None \
        else estimate_cam_offset(mcap, args.cam)
    cam_off_ns = int(round(cam_off * 1e9))
    print(f"[{label}] {os.path.basename(mcap)}: {N} targets, cam->lidar offset "
          f"{cam_off:.3f}s ({'given' if args.cam_offset is not None else 'auto'})")

    # single streaming pass, paired on header stamp (sensor time), NOT log_time.
    # lidar: keep the scan nearest the requested target.
    # camera: keep the K nearest (offset-aligned) frames to the target, then pick
    #         the one nearest the CHOSEN lidar -> minimises the lidar<->cam gap.
    KCAM = 5
    best_l = [None] * N            # (dist_to_target, data, schema, stamp)
    cams = [[] for _ in range(N)]  # [(dist, stamp_aligned, stamp_raw, data, schema)]
    with open(mcap, "rb") as f:
        for schema, channel, message in make_reader(f).iter_messages(topics=topics):
            if channel.topic == LIDAR_TOPIC:
                s = stamp_ns(message.data)
                for i, T in enumerate(targets):
                    d = abs(s - T)
                    if best_l[i] is None or d < best_l[i][0]:
                        best_l[i] = (d, message.data, schema, s)
            else:
                raw = stamp_ns(message.data)
                s = raw - cam_off_ns
                for i, T in enumerate(targets):
                    d = abs(s - T)
                    c = cams[i]
                    if len(c) < KCAM or d < c[-1][0]:
                        c.append((d, s, raw, message.data, schema))
                        c.sort(key=lambda e: e[0])
                        del c[KCAM:]

    fac = DecoderFactory() if decode else None
    rows = []
    seen_lidar = set()
    for i in range(N):
        if best_l[i] is None or not cams[i]:
            print(f"[{label}] [skip] target {i}: no lidar/camera message found")
            continue
        _, ld, lsch, l_s = best_l[i]
        _, c_al, c_raw, cd, csch = min(cams[i], key=lambda e: abs(e[1] - l_s))
        dup = " DUP-lidar" if l_s in seen_lidar else ""
        seen_lidar.add(l_s)
        xyz4 = bgr = None
        if decode:
            xyz4 = pc2_to_xyz4(fac.decoder_for("cdr", lsch)(ld), args.pcd_ch4)
            bgr = image_to_bgr(fac.decoder_for("cdr", csch)(cd))
        rows.append((label, (targets[i] - start) / 1e9, (l_s - start) / 1e9,
                     (c_al - start) / 1e9, (l_s - c_al) / 1e9, dup, l_s, c_raw,
                     xyz4, bgr))
    return rows


def mode_extract(args):
    import cv2
    # one or two bags; each timed on its own clock, all frames pooled into one dir
    bags = [("bag1", args.mcap, args.times, args.times_file)]
    if args.mcap2:
        bags.append(("bag2", args.mcap2, args.times2, args.times2_file))
    decode = not args.dry_run
    all_rows = []
    for label, mcap, times, tfile in bags:
        all_rows.extend(_extract_bag_rows(args, mcap, times, tfile, label, decode))

    outdir = args.outdir or os.path.join(
        os.path.dirname(args.mcap), f"lcecalib_{cam_tag(args.cam)}")

    # report (per-bag rel times; cam_s and off_ms on the aligned LiDAR clock)
    print(f"\n{'#':>3s} {'bag':>4s} {'want_s':>8s} {'lidar_s':>8s} {'cam_s':>8s} {'off_ms':>7s}")
    for k, r in enumerate(all_rows):
        print(f"{k:>3d} {r[0]:>4s} {r[1]:>8.3f} {r[2]:>8.3f} {r[3]:>8.3f} "
              f"{r[4]*1e3:>7.1f}{r[5]}")
    if args.dry_run:
        print("\n[dry-run] nothing written. Drop --dry-run to export.")
        return
    if not all_rows:
        sys.exit("no frames extracted")

    os.makedirs(os.path.join(outdir, "img"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "pcd"), exist_ok=True)
    csv = [("index,bag,want_rel_s,lidar_rel_s,cam_rel_s,lidar_minus_cam_ms,"
            "lidar_stamp_ns,cam_stamp_ns")]
    for k, r in enumerate(all_rows):   # continuous numbering across both bags
        stem = f"{k:06d}"
        write_pcd(os.path.join(outdir, "pcd", stem + ".pcd"), r[8])
        cv2.imwrite(os.path.join(outdir, "img", stem + ".png"), r[9])
        csv.append(f"{k},{r[0]},{r[1]:.3f},{r[2]:.3f},{r[3]:.3f},{r[4]*1e3:.1f},"
                   f"{r[6]},{r[7]}")
    with open(os.path.join(outdir, "frames_index.csv"), "w") as f:
        f.write("\n".join(csv) + "\n")
    print(f"\nwrote {len(all_rows)} pairs -> {outdir}/img/*.png + {outdir}/pcd/*.pcd")
    print(f"index map -> {outdir}/frames_index.csv")
    print("Next: drop an img/params.mat (K,D,borW,borH,numW,numH,pattern_size,"
          "num_data,...) and point LCECalib's data_path here.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["list", "preview", "extract"])
    ap.add_argument("mcap", nargs="?", default=DEFAULT_MCAP,
                    help="input .mcap bag path (default: %(default)s)")
    ap.add_argument("--cam", default=None,
                    help="camera: 'arenacam2', '2', or '/arenacam2/images'. "
                         "Default: inferred from the mcap filename "
                         "(lidar-camN -> arenacamN), else arenacam1")
    ap.add_argument("--mcap2", default=None,
                    help="extract: optional 2nd bag; its frames (--times2) are "
                         "appended to the same --outdir with continuous numbering")
    # preview
    ap.add_argument("--every", type=float, default=None,
                    help="preview: seconds between JPEGs; extract: uniform sample step")
    ap.add_argument("--preview-dir", default=None)
    ap.add_argument("--scale", type=float, default=0.5, help="preview downscale factor")
    ap.add_argument("--cam-offset", type=float, default=None,
                    help="seconds to subtract from camera header stamps to align "
                         "them to the LiDAR clock (default: auto-estimate)")
    # extract selection
    ap.add_argument("--times", default=None, help="comma/space list of times (sec)")
    ap.add_argument("--times-file", default=None, help="file with whitespace times")
    ap.add_argument("--times2", default=None, help="times (sec) for --mcap2")
    ap.add_argument("--times2-file", default=None, help="file with times for --mcap2")
    ap.add_argument("--abs", action="store_true",
                    help="treat --times as absolute unix seconds (default: rel to start)")
    ap.add_argument("--start", type=float, default=None, help="uniform sample start (s)")
    ap.add_argument("--end", type=float, default=None, help="uniform sample end (s)")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--pcd-ch4", choices=["ring", "intensity"], default="ring",
                    help="what to store in the PCD 'intensity' field: LiDAR ring "
                         "(default, required by LCECalib) or true reflectivity")
    ap.add_argument("--dry-run", action="store_true",
                    help="extract: show pairing table only, write nothing")
    args = ap.parse_args()
    if args.cam is None:   # infer camera number from the mcap filename (lidar-camN)
        m = re.search(r"cam(\d+)", os.path.basename(args.mcap))
        args.cam = cam_topic(m.group(1) if m else "arenacam1")
        print(f"[cam] using {args.cam} (inferred from {os.path.basename(args.mcap)})")
    else:
        args.cam = cam_topic(args.cam)
    if args.mode == "preview" and args.every is None:
        args.every = 2.0

    {"list": mode_list, "preview": mode_preview, "extract": mode_extract}[args.mode](args)


if __name__ == "__main__":
    main()
