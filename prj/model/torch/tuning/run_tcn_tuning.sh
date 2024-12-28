#!/usr/bin/bash
#SBATCH --job-name ObjectDetection3Dv5
#SBATCH --account DD-24-8
#SBATCH --partition qgpu
#SBATCH --gpus 1
#SBATCH --time 48:00:00
#SBATCH --error /mnt/proj2/dd-24-8/frustum_datasets/last/tcn/job.err
#SBATCH --output /mnt/proj2/dd-24-8/frustum_datasets/last/tcn/job.out
#SBATCH --gres=gpu:1
#SBATCH --nodes 1

module load CUDA/12.4.0
module load Python/3.10.4-GCCcore-11.3.0
source $HOME/venv_new/bin/activate

# Set GPUs to exclusive mode
# for gpu in {0..7}; do
#     nvidia-smi -i $gpu -c EXCLUSIVE_PROCESS
# done

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
export CUDA_VISIBLE_DEVICES=0
python $HOME/jane_street_forecasting/prj/model/torch/tuning/tcn.py \
	-output_dir /mnt/proj2/dd-24-8/frustum_datasets/last/tcn \
	-dataset_path /mnt/proj2/dd-24-8/frustum_datasets/last/dataset/train.parquet \
	-n_trials 100 \
    -study_name tcn \
	-storage sqlite:////mnt/proj2/dd-24-8/frustum_datasets/last/tuning.db \
    -n_gpus 1 \
	-n_gpus_per_trial 1 \
	-num_workers_per_dataloader 8