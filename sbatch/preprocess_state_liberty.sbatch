#!/bin/bash
#SBATCH --job-name=preproc_state_liberty
#SBATCH --account=mcity_project
#SBATCH --partition=mcity_project-l40s
#SBATCH --time=00:17:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j-E.log

set -euo pipefail

cd /home/$USER/drivestudio
source setup.sh

python tools/preprocess_mcity.py \
    --bag_path /scratch/mcity_project_root/mcity_project/$USER/mcity/may8-2026-downtown-rev-p7-take2_0.mcap \
    --start_s 6 --end_s 26 \
    --bag_path /scratch/mcity_project_root/mcity_project/$USER/mcity/may8-2026-downtown-p2_0.mcap \
    --start_s 80 --end_s 103 \
    --calib_root /home/$USER/drivestudio/calibration_files \
    --out_dir /scratch/mcity_project_root/mcity_project/$USER/drivestudio/data/mcity/processed/training/state_liberty \
    --hz 10

python tools/extract_sky_masks_hf.py \
    --data_root /scratch/mcity_project_root/mcity_project/$USER/drivestudio/data/mcity/processed/training \
    --scene_ids state_liberty
