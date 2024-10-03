#!/bin/bash
##Slurm sbatch options
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:volta:2
#SBATCH --exclusive -O

# Loading the required modules
source /etc/profile
module load anaconda/2023a

# Activate conda env
source activate train-clip

EXP_NAME=train
BACKBONE=clip-14
MODEL_NAME=ViT-L/14
CSV_FILE=/home/gridsan/manderson/vlm4rs/fmow/train_latest.csv
CAPTION_TYPE=0
BATCH_SIZE=125
NUM_WORKERS=16
WARMUP_PERCENT=0.1
MODEL_DIR=/home/gridsan/manderson/train-CLIP/run
MAX_EPOCHS=50
GPUS=1

python train_cp.py \
    --backbone ${BACKBONE} \
    --model_name ${MODEL_NAME} \
    --csv_file ${CSV_FILE} \
    --caption_type ${CAPTION_TYPE} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --warmup_percent ${WARMUP_PERCENT} \
    --model_dir ${MODEL_DIR}/${EXP_NAME} \
    --max_epochs ${MAX_EPOCHS} \
    --gpus ${GPUS} \
    --exp_name ${EXP_NAME}