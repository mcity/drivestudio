"""Sky-mask extraction using Hugging Face SegFormer (Cityscapes-1024 B5).

Drop-in replacement for the legacy datasets/tools/extract_masks.py sky-mask path.
Runs on H100 (sm_90) since it uses a modern PyTorch via transformers.

Outputs: {data_root}/{scene_id:03d}/sky_masks/{frame:03d}_{cam}.png
         (255 where sky, 0 elsewhere; same convention as the legacy script.)

Usage (in the drivestudio conda env):
    python tools/extract_sky_masks_hf.py \\
        --data_root /scratch/.../data/mcity/processed/training \\
        --scene_ids 0
"""
import argparse
import os
from glob import glob
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


CITYSCAPES_SKY_CLASS = 10  # same index as the legacy mmseg cityscapes palette


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--scene_ids", type=int, nargs="+", required=True)
    parser.add_argument("--model_name", default="nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
    parser.add_argument("--rgb_dirname", default="images")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    processor = SegformerImageProcessor.from_pretrained(args.model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_name).to(args.device).eval()

    for scene in args.scene_ids:
        scene_id = f"{scene:03d}"
        img_dir = Path(args.data_root) / scene_id / args.rgb_dirname
        out_dir = Path(args.data_root) / scene_id / "sky_masks"
        out_dir.mkdir(parents=True, exist_ok=True)

        flist = sorted(glob(str(img_dir / "*")))
        if not flist:
            print(f"[skip] {scene_id}: no images at {img_dir}")
            continue

        for i in tqdm(range(0, len(flist), args.batch_size), desc=f"scene[{scene_id}]"):
            batch_paths = flist[i:i + args.batch_size]
            pil_images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = processor(images=pil_images, return_tensors="pt").to(args.device)

            with torch.no_grad():
                outputs = model(**inputs)
            # logits: [B, C, H/4, W/4] - upsample to original size
            logits = outputs.logits
            target_sizes = [img.size[::-1] for img in pil_images]  # (H, W)
            seg = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
            for path, s in zip(batch_paths, seg):
                fbase = os.path.splitext(os.path.basename(path))[0]
                sky = (s.cpu().numpy() == CITYSCAPES_SKY_CLASS).astype(np.uint8) * 255
                Image.fromarray(sky).save(out_dir / f"{fbase}.png")

        print(f"[done] {scene_id}: {len(flist)} masks -> {out_dir}")


if __name__ == "__main__":
    main()
