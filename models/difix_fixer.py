import logging
import os
import random
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

logger = logging.getLogger()


class CameraPoseInterpolator:
    def __init__(self, rotation_weight: float = 1.0, translation_weight: float = 1.0):
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight

    def compute_pose_distance(self, pose1, pose2) -> float:
        t1, t2 = pose1[:3, 3], pose2[:3, 3]
        translation_dist = float(np.linalg.norm(t1 - t2))
        q1 = Rotation.from_matrix(pose1[:3, :3]).as_quat()
        q2 = Rotation.from_matrix(pose2[:3, :3]).as_quat()
        if np.dot(q1, q2) < 0:
            q2 = -q2
        cos_term = np.clip(2 * float(np.dot(q1, q2)) ** 2 - 1, -1.0, 1.0)
        rotation_dist = float(np.arccos(cos_term))
        return (self.translation_weight * translation_dist
                + self.rotation_weight * rotation_dist)

    def find_nearest_assignments(self, training_poses, testing_poses) -> List[int]:
        assignments = []
        for j in range(len(testing_poses)):
            distances = [
                self.compute_pose_distance(tp, testing_poses[j])
                for tp in training_poses
            ]
            assignments.append(int(np.argmin(distances)))
        return assignments

    def interpolate_rotation(self, R1, R2, t):
        q1 = Rotation.from_matrix(R1).as_quat()
        q2 = Rotation.from_matrix(R2).as_quat()
        if np.dot(q1, q2) < 0:
            q2 = -q2
        dot = float(np.clip(np.dot(q1, q2), -1.0, 1.0))
        theta = np.arccos(dot)
        if abs(theta) < 1e-6:
            q = (1 - t) * q1 + t * q2
        else:
            q = (np.sin((1 - t) * theta) * q1 + np.sin(t * theta) * q2) / np.sin(theta)
        q = q / np.linalg.norm(q)
        return Rotation.from_quat(q).as_matrix()

    def shift_poses(self, training_poses, testing_poses, distance: float = 0.5):
        assignments = self.find_nearest_assignments(training_poses, testing_poses)
        novel_poses = []
        for test_idx, train_idx in enumerate(assignments):
            train_pose = training_poses[train_idx]
            test_pose = testing_poses[test_idx]

            if self.compute_pose_distance(train_pose, test_pose) <= distance:
                novel_poses.append(test_pose)
                continue

            t1, t2 = train_pose[:3, 3], test_pose[:3, 3]
            direction = t2 - t1
            norm = float(np.linalg.norm(direction))
            if norm > 1e-6:
                new_t = t1 + (direction / norm) * distance
            else:
                new_t = t2

            if (np.dot(new_t - t1, t2 - t1) <= 0
                    or np.linalg.norm(new_t - t2) <= distance):
                new_t = t2

            R1 = train_pose[:3, :3]
            R2 = test_pose[:3, :3]
            if norm > 1e-6:
                R_new = self.interpolate_rotation(R1, R2, min(distance / norm, 1.0))
            else:
                R_new = R2

            pose = np.eye(4)
            pose[:3, :3] = R_new
            pose[:3, 3] = new_t
            novel_poses.append(pose)

        return np.array(novel_poses)


class DifixFixer:
    """Progressive 3D update fixer using a frozen Difix3D diffusion model.

    Mirrors third_party/Difix3D/examples/gsplat/simple_trainer_difix3d.py for
    the drivestudio background-only configuration. At each fix step it renders
    a batch of novel poses, runs Difix on each (with the nearest training image
    as reference), and stores the cleaned image + the novel-frame
    image_infos/cam_infos in an in-memory pool. The training loop samples from
    this pool to supervise the splat with the cleaned image as pseudo-GT.
    """

    def __init__(self, cfg, dataset, trainer, log_dir: str):
        from pipeline_difix import DifixPipeline

        self.cfg = cfg
        self.dataset = dataset
        self.trainer = trainer
        self.log_dir = log_dir
        self.device = getattr(trainer, "device", torch.device("cuda"))

        model_id = cfg.get("model_id", "nvidia/difix_ref")
        logger.info(f"Loading Difix pipeline: {model_id}")
        self.pipe = DifixPipeline.from_pretrained(model_id, trust_remote_code=True)
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.to("cuda")

        ps = dataset.pixel_source
        self.front_cam_id = ps.camera_list[0]
        front_cam = ps.camera_data[self.front_cam_id]
        cam_to_worlds = front_cam.cam_to_worlds.detach().cpu().numpy().astype(np.float64)
        all_img_paths = [str(p) for p in getattr(front_cam, "img_filepaths", [])]

        train_ts = np.asarray(dataset.train_timesteps, dtype=int)
        test_ts = np.asarray(dataset.test_timesteps, dtype=int)

        if len(test_ts) > 0:
            self.training_poses = cam_to_worlds[train_ts]
            self.target_poses = cam_to_worlds[test_ts]
            self.train_image_paths = (
                [all_img_paths[i] for i in train_ts] if all_img_paths else []
            )
            logger.info(
                f"[Difix] using {len(test_ts)} held-out val poses as targets "
                f"(vs {len(train_ts)} training poses)"
            )
        else:
            traj_types = list(cfg.get("traj_types", ["front_center_interp"]))
            num_novel_poses = int(cfg.get("num_novel_poses", 60))
            traj_dict = dataset.get_novel_render_traj(
                traj_types=traj_types, target_frames=num_novel_poses,
            )
            targets = []
            for v in traj_dict.values():
                arr = v if isinstance(v, np.ndarray) else v.detach().cpu().numpy()
                targets.append(arr)
            self.target_poses = np.concatenate(targets, axis=0).astype(np.float64)
            self.training_poses = cam_to_worlds
            self.train_image_paths = all_img_paths
            logger.warning(
                "[Difix] no held-out val poses (test_image_stride=0); "
                "falling back to synthetic trajectories"
            )

        self.interp = CameraPoseInterpolator(rotation_weight=1.0, translation_weight=1.0)
        self.current_poses = self.training_poses.copy()

        self.pool: List[Dict] = []
        self.fix_steps_set = set(int(s) for s in cfg.get("fix_steps", []))

        self.save_intermediate = bool(cfg.get("save_intermediate_pngs", True))
        os.makedirs(os.path.join(log_dir, "difix"), exist_ok=True)

    def should_fix(self, step: int) -> bool:
        return step in self.fix_steps_set

    @torch.no_grad()
    def fix(self, step: int) -> None:
        cfg = self.cfg
        dataset = self.dataset
        trainer = self.trainer

        distance = float(cfg.get("shift_distance", 0.5))
        new_poses = self.interp.shift_poses(
            self.current_poses, self.target_poses, distance=distance,
        )

        traj_tensor = torch.from_numpy(new_poses).float().to(self.device)
        render_data = dataset.prepare_novel_view_render_data(traj_tensor)
        ref_indices = self.interp.find_nearest_assignments(self.training_poses, new_poses)

        pred_dir = os.path.join(self.log_dir, "difix", f"{step}", "pred")
        fixed_dir = os.path.join(self.log_dir, "difix", f"{step}", "fixed")
        if self.save_intermediate:
            os.makedirs(pred_dir, exist_ok=True)
            os.makedirs(fixed_dir, exist_ok=True)

        was_training = trainer.training if hasattr(trainer, "training") else True
        trainer.set_eval()
        new_entries: List[Dict] = []
        logger.info(f"[Difix] step {step}: rendering and fixing {len(render_data)} novel views")
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
                image_infos=image_infos,
                camera_infos=cam_infos,
                novel_view=True,
            )
            rgb = outputs["rgb"].clamp(0.0, 1.0)
            H, W = rgb.shape[0], rgb.shape[1]

            pred_np = (rgb.detach().cpu().numpy() * 255).astype(np.uint8)
            pred_pil = Image.fromarray(pred_np, mode="RGB")
            if self.save_intermediate:
                pred_pil.save(os.path.join(pred_dir, f"{i:04d}.png"))

            ref_pil = None
            if self.train_image_paths:
                ref_path = self.train_image_paths[
                    ref_indices[i] % len(self.train_image_paths)
                ]
                if os.path.exists(ref_path):
                    ref_pil = Image.open(ref_path).convert("RGB")
                    if ref_pil.size != (W, H):
                        ref_pil = ref_pil.resize((W, H), Image.LANCZOS)

            pipe_kwargs = dict(
                image=pred_pil,
                num_inference_steps=1,
                timesteps=[199],
                guidance_scale=0.0,
            )
            if ref_pil is not None:
                pipe_kwargs["ref_image"] = ref_pil
            fixed_pil = self.pipe("remove degradation", **pipe_kwargs).images[0]
            if fixed_pil.size != (W, H):
                fixed_pil = fixed_pil.resize((W, H), Image.LANCZOS)
            if self.save_intermediate:
                fixed_pil.save(os.path.join(fixed_dir, f"{i:04d}.png"))

            fixed_np = np.asarray(fixed_pil).astype(np.float32) / 255.0
            pixels_target = torch.from_numpy(fixed_np).to(self.device)

            new_entries.append({
                "image_infos": frame["image_infos"],
                "cam_infos": frame["cam_infos"],
                "pixels_target": pixels_target,
            })

        self.pool = new_entries
        self.current_poses = new_poses
        if was_training:
            trainer.set_train()
        logger.info(f"[Difix] step {step}: pool size = {len(self.pool)} (replaced)")

    def sample(self) -> Optional[Dict]:
        if not self.pool:
            return None
        return random.choice(self.pool)
