cd /home/$USER/drivestudio
export PYTHONPATH=$(pwd)

scene_idx='"000"'          # mcity processed scene
output_root=/scratch/mcity_project_root/mcity_project/$USER/drivestudio/output
project=mcity
expname=state_liberty

# For H100 - mcity background-only
TORCH_CUDA_ARCH_LIST="9.0" python tools/train.py \
    --config_file configs/omnire_bg_only.yaml \
    --output_root $output_root \
    --project $project \
    --run_name $expname \
    dataset=mcity/6cams \
    data.scene_idx=$scene_idx \
    data.start_timestep=0 \
    data.end_timestep=-1 \
    logging.saveckpt_freq=5000

# For L40S
# CUDA_LAUNCH_BLOCKING=1 TORCH_CUDA_ARCH_LIST="8.9" python tools/train.py \
#     --config_file configs/omnire.yaml \
#     --output_root $output_root \
#     --project $project \
#     --run_name $expname \
#     dataset=waymo/3cams \
#     data.scene_idx=$scene_idx \
#     data.start_timestep=0 \
#     data.end_timestep=-1 \
#     logging.saveckpt_freq=5000