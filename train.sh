# VITONHD base
export CUDA_VISIBLE_DEVICES=0,1,2,3
#
conda activate DGIST_VTON

# Base Training, 1000 epochs
python train.py \
 --config_name DGIST \
 --transform_size shiftscale hflip \
 --transform_color hsv bright_contrast \
 --save_name Base \
 --max_epochs 1000 \
 --save_every_n_epochs 100
 #--resume_path logs/20240326_Base/models/[Train]_[epoch=999]_[train_loss_epoch=0.0404].ckpt\

# lr = 1e-5, 300 epochs
# python train.py \
#  --config_name VITONHD \
#  --transform_size shiftscale3 hflip \
#  --transform_color hsv bright_contrast \
#  --save_name Base_test \
#  --resume_path "logs/20240321_Base_test/models/{name}.ckpt"\
#  --max_epochs 300 \
#  --save_every_n_epochs 100 \
#  --learning_rate 0.00001 \

# VITONHD ATVloss
# CUDA_VISIBLE_DEVICES=0 python train.py \
#  --config_name VITONHD \
#  --transform_size shiftscale hflip \
#  --transform_color hsv bright_contrast \
#  --use_atv_loss \
#  --resume_path "logs/20240326_Base/models/[Train]_[epoch=999]_[train_loss_epoch=0.0404].ckpt" \
#  --save_name ATVloss \
#  --save_every_n_epochs 100

conda deactivate
