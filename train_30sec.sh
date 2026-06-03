cd /home/$USER/drivestudio
export PYTHONPATH=$(pwd)
export PYTHONPATH=$PYTHONPATH:/home/$USER/drivestudio/third_party/Difix3D/src
export PYTHONNOUSERSITE=1   # prevents ~/.local from shadowing the conda env's huggingface_hub==0.25.1

scene_idx='"30sec"'   # mcity processed scene (subdir name under data/mcity/processed/training)
output_root=/scratch/mcity_project_root/mcity_project/$USER/drivestudio/output
project=mcity
expname=30sec_no_difix

# For H100 - mcity background-only with Difix3D progressive update
TORCH_CUDA_ARCH_LIST="9.0" python tools/train.py \
    --config_file configs/omnire_bg_only.yaml \
    --output_root $output_root \
    --project $project \
    --run_name $expname \
    dataset=mcity/6cams \
    data.scene_idx=$scene_idx \
    data.start_timestep=0 \
    data.end_timestep=-1 \
    logging.saveckpt_freq=5000 \
    difix.enabled=false
