CUDA_VISIBLE_DEVICES=3 python inference.py \
 --config_path ./configs/DGIST.yaml \
 --batch_size 16 \
 --model_load_path ./logs/20250131_Base/models/[Train]_[epoch=399]_[train_loss_epoch=0.0414].ckpt \
 --save_dir ./inference/20250131_Base/399epoch/repaint \
 --data_root_dir ./DATA/VITON-HD \
 --repaint

# #### paired
# CUDA_VISIBLE_DEVICES=4 python inference.py \
#  --config_path ./configs/VITONHD.yaml \
#  --batch_size 4 \
#  --model_load_path <model weight path> \
#  --save_dir <save directory>

# #### unpaired
# CUDA_VISIBLE_DEVICES=4 python inference.py \
#  --config_path ./configs/VITONHD.yaml \
#  --batch_size 4 \
#  --model_load_path <model weight path> \
#  --unpair \
#  --save_dir <save directory>

# #### paired repaint
# CUDA_VISIBLE_DEVICES=4 python inference.py \
#  --config_path ./configs/VITONHD.yaml \
#  --batch_size 4 \
#  --model_load_path <model weight path>t \
#  --repaint \
#  --save_dir <save directory>

# #### unpaired repaint
# CUDA_VISIBLE_DEVICES=4 python inference.py \
#  --config_path ./configs/VITONHD.yaml \
#  --batch_size 4 \
#  --model_load_path <model weight path> \
#  --unpair \
#  --repaint \
#  --save_dir <save directory>