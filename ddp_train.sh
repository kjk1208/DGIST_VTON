#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

eval "$(conda shell.bash hook)"
conda activate vton

# Base Training, 1000 epochs
torchrun --nproc_per_node=2 train_ddp.py \
 --config_name DGIST \
 --transform_size shiftscale hflip \
 --transform_color hsv bright_contrast \
 --save_name Base \
 --resume_path ./ckpts/updated_initial_checkpoint.ckpt \
 --max_epochs 1000 \
 --save_every_n_epochs 200 \
 --no_strict_load \
 --batch_size 32 \
 --learning_rate 6e-5 \
 --strategy ddp


 #--resume_path /data/StableVITON/logs/20240326_Base/models/[Train]_[epoch=999]_[train_loss_epoch=0.0404].ckpt\