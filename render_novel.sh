# In drivestudio root, source setup.sh first
export PYTHONNOUSERSITE=1
export PYTHONPATH=$(pwd):/home/$USER/drivestudio/third_party/Difix3D/src

CKPT=/scratch/mcity_project_root/mcity_project/$USER/drivestudio/output/mcity/state_liberty_difix/checkpoint_final.pth
# replace 30000 with whatever your final step was — check the ls above

python tools/eval.py \
    --resume_from $CKPT \
    render.render_full=false \
    render.render_test=false
