#CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --nproc_per_node=2 --master_port 19961229 \
#  tools/train.py configs/segformer/04_22_segformer_mit-b3_2xb8-40k_agriculture-vision-RGBN-512x512.py

#export CUDA_VISIBLE_DEVICES=0,2
#bash tools/dist_train.sh configs/segformer/04_22_segformer_mit-b3_2xb8-40k_agriculture-vision-RGBN-512x512.py 2


export CUDA_VISIBLE_DEVICES=4,5,6,7
bash tools/dist_train.sh \
  configs/segformer/06_02_segformer_mit-b5_4xb4-80k_agriculture-vision-RGBN-512x512_ACWLoss_mosaicx4_rc0.75-1.5_NoiseBlur_ClassMix.py \
  4

bash tools/dist_train.sh \
  configs/upernet/05_16_upernet_mit-b5_4xb4-80k_agrivisoin-512x512-mosaic4x.py \
  4

bash tools/dist_train.sh \
  configs/deeplabv3plus/05_17_deeplabv3plus_mit-b5_4xb4-80k_agrivision-512x512-mosaic4x.py \
  4

bash tools/dist_train.sh \
  configs/swin/05_17_swin-large-patch4-window7-in22k-pre_deeplabv3plus_4xb4-80k_agrivision-512x512.py \
  4

export CUDA_VISIBLE_DEVICES=6,7
bash tools/dist_train.sh \
  configs/deeplabv3plus/05_16_deeplabv3plus_mit-b3_2xb8-80k_agrivision-512x512-mosaic4x.py \
  2

bash tools/dist_train.sh \
  cconfigs/upernet/05_16_upernet_mit-b3_2xb8-80k_agrivisoin-512x512-mosaic4x.py \
  2
