import os
from PIL import Image
import cv2
import torch
from tqdm import tqdm
from argparse import ArgumentParser

from segment_anything import (SamAutomaticMaskGenerator, SamPredictor,
                              sam_model_registry)

if __name__ == '__main__':
    
    parser = ArgumentParser(description="SAM feature extracting params")
    
    parser.add_argument("--image_root", default='data/nuscenes/samples/CAM_BACK', type=str)
    parser.add_argument("--sam_checkpoint_path", default="ckpts/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--sam_arch", default="vit_h", type=str)
    parser.add_argument("--first_half", action="store_true",default=False)

    args = parser.parse_args()
    
    print("Initializing SAM...")
    model_type = args.sam_arch
    sam = sam_model_registry[model_type](checkpoint=args.sam_checkpoint_path).to('cuda')
    predictor = SamPredictor(sam)
    
    # IMAGE_DIR = os.path.join(args.image_root, 'images')
    IMAGE_DIR = args.image_root
    assert os.path.exists(IMAGE_DIR) and "Please specify a valid image root"
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(args.image_root)), 'SAM_features_', args.image_root.split('/')[-1])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    pathes = os.listdir(IMAGE_DIR)
    num = len(pathes)//2
    first_half = args.first_half
    if first_half:
        pathes = pathes[:num]
        print("processing the first half")
    else:
        pathes = pathes[num:]  
        print("processing the second half")  
    print("Extracting SAM segment everything features...")


    print("Extracting features...")
    for path in tqdm(pathes):
        name = path.split('.')[0]
        if os.path.exists(os.path.join(OUTPUT_DIR, name+'.pt')):
            continue
        img = cv2.imread(os.path.join(IMAGE_DIR, path))
        # img = cv2.resize(img,dsize=(1024,1024),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
        predictor.set_image(img)
        features = predictor.features
        torch.save(features, os.path.join(OUTPUT_DIR, name+'.pt'))