module load python3.9-anaconda/2021.11
module load cuda/12.6.3
module load tmux/3.3a
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate
conda activate drivestudio
export TORCH_CUDA_ARCH_LIST="8.9;9.0" 
export FORCE_CUDA=1 
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))