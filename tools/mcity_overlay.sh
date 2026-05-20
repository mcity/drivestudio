scene_dir=/scratch/mcity_project_root/mcity_project/billhong/drivestudio/data/mcity/processed/training/state_liberty
out_dir=~/drivestudio/overlay_results

for f in 0 50 100 150 195; do
  python tools/verify_mcity_overlay.py \
    --scene_dir "$scene_dir" \
    --frame $f --out_dir "$out_dir"
done
