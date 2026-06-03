#!/bin/bash
#SBATCH --job-name=train_state_liberty_difix
#SBATCH --account=mcity_project
#SBATCH --partition=mcity_project-l40s
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j-E.log

set -euo pipefail

cd /home/$USER/drivestudio
source setup.sh
# If Difix3D was installed in a separate conda env, activate it here instead:
#   conda activate difix3d_env

bash train_state_liberty_difix.sh
