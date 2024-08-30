# export PYTHONPATH=$(pwd)
# CUDA_VISIBLE_DEVICES=0 python cldm/cldm.py

export CUDA_VISIBLE_DEVICES=0 

conda activate StableVITON

# Base Training, 1000 epochs
python network_train.py \
 --config_name DGIST \
 --transform_size shiftscale hflip \
 --transform_color hsv bright_contrast \
 --save_name Base \
 --resume_path ./ckpts/updated_initial_checkpoint.ckpt \
 --max_epochs 1000 \
 --save_every_n_epochs 100 \
 --no_strict_load \
 --batch_size 24 \
 --learning_rate 1e-6

 #--resume_path ./ckpts/updated_initial_checkpoint.ckpt \

# 20240707 : PBE weight copy to controlnet (controlnet can update)
# 20240717 : controlnet can not update (only attention)
# ??? : spatial attention update

 #--resume_path /data/StableVITON/logs/20240326_Base/models/[Train]_[epoch=999]_[train_loss_epoch=0.0404].ckpt\