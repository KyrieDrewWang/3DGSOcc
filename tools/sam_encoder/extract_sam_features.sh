#!/bin/sh

for dir in data/nuscenes/samples/CAM_BACK_LEFT # data/nuscenes/samples/CAM_BACK_LEFT data/nuscenes/samples/CAM_BACK_RIGHT
do
echo "processing: ${dir##*/}"
python \
tools/sam_encoder/extract_sam_features.py \
--sam_checkpoint_path ckpts/sam_vit_h_4b8939.pth \
--sam_arch vit_h \
--image_root ${dir}
done
# for dir in data/nuscenes/samples/CAM_FRONT data/nuscenes/samples/CAM_FRONT_LEFT data/nuscenes/samples/CAM_FRONT_RIGHT data/nuscenes/samples/CAM_BACK data/nuscenes/samples/CAM_BACK_LEFT data/nuscenes/samples/CAM_BACK_RIGHT
# do
# echo "processing: ${dir##*/}"
# CUDA_VISIBLE_DEVICES=0 python \
# tools/sam_encoder/extract_sam_features.py \
# --sam_checkpoint_path ckpts/sam_vit_h_4b8939.pth \
# --sam_arch vit_h \
# --image_root ${dir}
# done