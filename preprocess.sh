source setup.sh

# New split-bag recordings: cameras (camA=cams1-3, camB=cams4-6) and the lidar/
# INS/tf bag are recorded simultaneously. Frames are paired by PTP header stamp;
# camera headers are TAI, +37s ahead of the lidar/INS/tf UTC clock (see
# --cam_offset_s, default 37). Point --lidar_bag/--cam_bag at the desired scene.
ZONE=/scratch/mcity_project_root/mcity_project/$USER/mcity_nurec/data/july2-2026/zone_2

python tools/preprocess_mcity.py \
    --lidar_bag $ZONE/lidar_ins_tf/lidar_ins_tf_0.mcap \
    --cam_bag   $ZONE/camA/camA_0.mcap \
    --cam_bag   $ZONE/camB/camB_0.mcap \
    --calib_root /home/$USER/drivestudio/calibration_files \
    --out_dir /scratch/mcity_project_root/mcity_project/$USER/drivestudio/data/mcity/processed/training/zone_2_turn \
    --start_s 55 --end_s 85 --hz 10

python tools/extract_sky_masks_hf.py \
    --data_root /scratch/mcity_project_root/mcity_project/$USER/drivestudio/data/mcity/processed/training \
    --scene_ids zone_2_turn
