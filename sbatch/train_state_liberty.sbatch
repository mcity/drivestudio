#!/bin/bash
#SBATCH --job-name=preproc_state_liberty
#SBATCH --account=mcity_project
#SBATCH --partition=mcity_project-l40s
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j-E.log

set -euo pipefail

cd /home/$USER/drivestudio
source setup.sh

bash train_state_liberty.sh
