cd /home/billhong/drivestudio
export PYTHONPATH=$(pwd)

scene_idx='"023"'          # whichever scene you're training on
output_root=/scratch/mcity_project_root/mcity_project/billhong/drivestudio/output
project=waymo
expname=test_run

# For H100
TORCH_CUDA_ARCH_LIST="9.0" python tools/train.py \
    --config_file configs/omnire.yaml \
    --output_root $output_root \
    --project $project \
    --run_name $expname \
    dataset=waymo/3cams \
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