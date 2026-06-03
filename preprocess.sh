source setup.sh

python tools/preprocess_mcity.py \
    --bag_path /scratch/mcity_project_root/mcity_project/$USER/mcity/may8-2026-downtown-p2_0.mcap \
    --start_s 0 --end_s 30 \
    --calib_root /home/$USER/drivestudio/calibration_files \
    --out_dir /scratch/mcity_project_root/mcity_project/$USER/drivestudio/data/mcity/processed/training/30sec \
    --hz 10

python tools/extract_sky_masks_hf.py \
    --data_root /scratch/mcity_project_root/mcity_project/$USER/drivestudio/data/mcity/processed/training \
    --scene_ids 30sec