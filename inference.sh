#### paired
CUDA_VISIBLE_DEVICES=0 python inference.py \
 --config_path ./configs/DGIST.yaml \
 --batch_size 50 \
 --model_load_path /data/kjk1208/DGIST_VTON/ControlnetDecoder_CrossAttention/20241014_Base/models/[Train]_[epoch=999]_[train_loss_epoch=0.0513].ckpt \
 --save_dir ./inference/20241014_Base/ \
 --data_root_dir ./DATA/VITON-HD \
 
 #--repaint
 #--model_load_path /home/kjk/DGIST_VTON/logs/20241014_Base/models/[Train]_[epoch=999]_[train_loss_epoch=0.0513].ckpt \
 #--model_load_path /home/kjk/DGIST_VTON/logs/20240707_Base/[Train]_[epoch=999]_[train_loss_epoch=0.0504].ckpt \


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