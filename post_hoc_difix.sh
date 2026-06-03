source setup.sh
export PYTHONNOUSERSITE=1
export PYTHONPATH=$(pwd):/home/$USER/drivestudio/third_party/Difix3D/src

run_name=30sec

CKPT=/scratch/mcity_project_root/mcity_project/$USER/drivestudio/output/mcity/$run_name/checkpoint_final.pth

python tools/difix_refine.py --resume_from "$CKPT"
