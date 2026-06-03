"""Post-hoc Difix3D+ refinement of a trained splat's novel-view renders.

Loads a trained checkpoint, renders novel-view trajectories from the config's
`render.render_novel.traj_types`, and runs each frame through the Difix3D
diffusion model. The splat itself is NOT modified — this is purely render-time
2D cleanup.

Usage:
    python tools/difix_refine.py --resume_from <path/to/checkpoint_final.pth>
"""
import argparse
import logging
import os
from typing import List, Tuple

import imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from datasets.driving_dataset import DrivingDataset
from models.difix_fixer import CameraPoseInterpolator
from utils.misc import import_str

logger = logging.getLogger("difix_refine")


def build_trainer_and_dataset(args):
    log_dir = os.path.dirname(args.resume_from)
    cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
    if args.opts:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device,
    )
    trainer.resume_from_checkpoint(ckpt_path=args.resume_from, load_only_model=True)
    trainer.set_eval()
    return cfg, dataset, trainer, log_dir


def build_difix(model_id: str):
    from pipeline_difix import DifixPipeline
    logger.info(f"Loading Difix pipeline: {model_id}")
    pipe = DifixPipeline.from_pretrained(model_id, trust_remote_code=True)
    pipe.set_progress_bar_config(disable=True)
    pipe.to("cuda")
    return pipe


def collect_train_refs(dataset) -> Tuple[np.ndarray, List[str]]:
    ps = dataset.pixel_source
    front_cam = ps.camera_data[ps.camera_list[0]]
    c2w = front_cam.cam_to_worlds.detach().cpu().numpy().astype(np.float64)
    img_paths = [str(p) for p in getattr(front_cam, "img_filepaths", [])]
    train_ts = np.asarray(dataset.train_timesteps, dtype=int)
    if len(train_ts) > 0 and img_paths:
        return c2w[train_ts], [img_paths[i] for i in train_ts]
    return c2w, img_paths


def render_and_clean(
    trainer, dataset, traj_type, traj, pipe,
    train_poses, train_image_paths, use_ref, output_dir, fps,
):
    pred_dir = os.path.join(output_dir, "pred")
    fixed_dir = os.path.join(output_dir, "fixed")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(fixed_dir, exist_ok=True)

    interp = CameraPoseInterpolator()
    traj_np = traj.detach().cpu().numpy().astype(np.float64) \
        if isinstance(traj, torch.Tensor) else np.asarray(traj).astype(np.float64)

    ref_indices = (
        interp.find_nearest_assignments(train_poses, traj_np)
        if use_ref and train_image_paths else []
    )

    render_data = dataset.prepare_novel_view_render_data(
        traj if isinstance(traj, torch.Tensor)
        else torch.from_numpy(traj_np).float().cuda()
    )

    pred_writer = imageio.get_writer(os.path.join(output_dir, "pred.mp4"), fps=fps)
    fixed_writer = imageio.get_writer(os.path.join(output_dir, "fixed.mp4"), fps=fps)

    logger.info(f"[{traj_type}] rendering+cleaning {len(render_data)} frames")
    with torch.no_grad():
        for i, frame in enumerate(render_data):
            image_infos = {
                k: (v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in frame["image_infos"].items()
            }
            cam_infos = {
                k: (v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in frame["cam_infos"].items()
            }

            outputs = trainer(
                image_infos=image_infos, camera_infos=cam_infos, novel_view=True,
            )
            rgb = outputs["rgb"].clamp(0.0, 1.0)
            H, W = rgb.shape[0], rgb.shape[1]

            pred_np = (rgb.detach().cpu().numpy() * 255).astype(np.uint8)
            pred_pil = Image.fromarray(pred_np, mode="RGB")
            pred_pil.save(os.path.join(pred_dir, f"{i:04d}.png"))
            pred_writer.append_data(pred_np)

            ref_pil = None
            if use_ref and ref_indices:
                ref_path = train_image_paths[ref_indices[i] % len(train_image_paths)]
                if os.path.exists(ref_path):
                    ref_pil = Image.open(ref_path).convert("RGB")
                    if ref_pil.size != (W, H):
                        ref_pil = ref_pil.resize((W, H), Image.LANCZOS)

            pipe_kwargs = dict(
                image=pred_pil, num_inference_steps=1,
                timesteps=[199], guidance_scale=0.0,
            )
            if ref_pil is not None:
                pipe_kwargs["ref_image"] = ref_pil
            fixed_pil = pipe("remove degradation", **pipe_kwargs).images[0]
            if fixed_pil.size != (W, H):
                fixed_pil = fixed_pil.resize((W, H), Image.LANCZOS)
            fixed_pil.save(os.path.join(fixed_dir, f"{i:04d}.png"))
            fixed_writer.append_data(np.asarray(fixed_pil))

    pred_writer.close()
    fixed_writer.close()
    logger.info(f"[{traj_type}] done -> {output_dir}")


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg, dataset, trainer, log_dir = build_trainer_and_dataset(args)
    pipe = build_difix(args.model_id)
    train_poses, train_image_paths = collect_train_refs(dataset)

    render_novel_cfg = cfg.render.get("render_novel", None)
    if render_novel_cfg is None:
        raise RuntimeError("cfg.render.render_novel must be set in the run config")

    traj_types = list(render_novel_cfg.traj_types)
    target_frames = int(render_novel_cfg.get("frames", dataset.frame_num))
    fps = int(render_novel_cfg.get("fps", cfg.render.fps))

    traj_dict = dataset.get_novel_render_traj(
        traj_types=traj_types, target_frames=target_frames,
    )

    output_root = os.path.join(log_dir, "difix_refine")
    os.makedirs(output_root, exist_ok=True)

    for traj_type, traj in traj_dict.items():
        out = os.path.join(output_root, traj_type)
        render_and_clean(
            trainer, dataset, traj_type, traj, pipe,
            train_poses, train_image_paths, args.use_ref, out, fps,
        )

    logger.info(f"All done. Outputs under: {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Difix3D+ post-render refinement")
    parser.add_argument("--resume_from", required=True, type=str,
                        help="path to splat checkpoint .pth")
    parser.add_argument("--model_id", default="nvidia/difix_ref", type=str,
                        help="HF model id (use nvidia/difix for no-reference mode)")
    parser.add_argument("--use_ref", action="store_true", default=True,
                        help="use nearest training image as Difix reference (default: True)")
    parser.add_argument("--no_ref", action="store_false", dest="use_ref",
                        help="disable reference image conditioning")
    parser.add_argument("opts", nargs=argparse.REMAINDER, default=None,
                        help="OmegaConf overrides to the loaded config")
    args = parser.parse_args()
    main(args)
