for dir in data/nuscenes/samples/CAM_BACK data/nuscenes/samples/CAM_BACK_LEFT data/nuscenes/samples/CAM_BACK_RIGHT
do
echo "processing: ${dir##*/}"
CUDA_VISIBLE_DEVICES=5 python \
tools/sam_encoder/export_image_embeddings.py \
--checkpoint tools/sam_encoder/checkpoints/sam_vit_h_4b8939.pth \
--model-type vit_h \
--input ${dir} \
--output data/nuscenes/SAM_embeddings/${dir##*/}
done


# for dir in data/nuscenes/samples/CAM_BACK data/nuscenes/samples/CAM_BACK_LEFT data/nuscenes/samples/CAM_BACK_RIGHT data/nuscenes/samples/CAM_FRONT data/nuscenes/samples/CAM_FRONT_LEFT data/nuscenes/samples/CAM_FRONT_RIGHT 
# do
# echo "processing: ${dir##*/}"
# CUDA_VISIBLE_DEVICES=8 python \
# tools/sam_encoder/export_image_embeddings.py \
# --checkpoint tools/sam_encoder/checkpoints/sam_vit_h_4b8939.pth \
# --model-type vit_h \
# --input ${dir} \
# --output data/nuscenes/SAM_embeddings/${dir##*/}
# done