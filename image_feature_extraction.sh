#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=LongJob
#SBATCH --gres=gpu:1
#SBATCH --mem=34000  # memory in Mb
#SBATCH --time=0-31:00:00


export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}

export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

mkdir -p ${TMP}/datasets/
export DATASET_DIR=${TMP}/datasets/
# Activate the relevant virtual environment:


source /home/${STUDENT_ID}/miniconda3/bin/activate mlp
cd ..

data_dir = "/home/s1885778/nrl/dataset/avito-duplicate-ads-detection/Images"
target_dir = "/disk/scratch/dataset/"
mkdir -p ${target_dir}
rsync -ua --progress data_dir target_dir

python preprocessing/image_feature_extraction.py --batch_size 64
                                                 --dataset_name "/disk/scratch/datasets/avito-duplicate-ads-detection/Images"
                                                 --use_gpu "True" --gpu_id "0"