import os
from PIL import Image
import cv2
import torch
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
from segment_anything import (SamAutomaticMaskGenerator, SamPredictor,
                              sam_model_registry)
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255/255, 144/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=50):
    pos_points = coords[labels!=0]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=0.3)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=0.3)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    



def load_depth(img_file_path, gt_path):
    file_name = os.path.split(img_file_path)[-1]
    cam_depth = np.fromfile(os.path.join(gt_path, f'{file_name}.bin'),
        dtype=np.float32,
        count=-1).reshape(-1, 3)
    
    coords = cam_depth[:, :2].astype(np.int16)
    depth_label = cam_depth[:,2]
    return coords, depth_label  

def load_seg_label(img_file_path, gt_path, img_size=[900,1600], mode='lidarseg'):
    if mode=='lidarseg':  # proj lidarseg to img
        coor, seg_label = load_depth(img_file_path, gt_path)
        seg_map = np.zeros(img_size)
        seg_map[coor[:, 1],coor[:, 0]] = seg_label
    else:
        file_name = os.path.join(gt_path, f'{os.path.split(img_file_path)[-1]}.npy')
        seg_map = np.load(file_name)
    return seg_map

if __name__ == '__main__':
    
    parser = ArgumentParser(description="SAM feature extracting params")
    
    parser.add_argument("--image_root", default='data/nuscenes/samples/CAM_FRONT', type=str)
    parser.add_argument("--sam_checkpoint_path", default="ckpts/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--sam_arch", default="vit_h", type=str)
    parser.add_argument("--depth_gt_path", default="data/nuscenes/depth_gt", type=str)
    parser.add_argument("--semantic_gt_path", default="data/nuscenes/seg_gt_lidarseg", type=str)
    parser.add_argument("--first_half", action="store_true",default=False)
    args = parser.parse_args()
    
    print("Initializing SAM...")
    model_type = args.sam_arch
    sam = sam_model_registry[model_type](checkpoint=args.sam_checkpoint_path).to('cuda')
    predictor = SamPredictor(sam)
    
    # IMAGE_DIR = os.path.join(args.image_root, 'images')
    IMAGE_DIR = args.image_root
    assert os.path.exists(IMAGE_DIR) and "Please specify a valid image root"
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(args.image_root)), 'SAM_prompt_mask_vis_', args.image_root.split('/')[-1])
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
    print("Extracting SAM segment everything masks...")


    print("Extracting features...")
    for path in tqdm(pathes[10000:]):
        name = path.split('.')[0]
        if os.path.exists(os.path.join(OUTPUT_DIR, name+'.pt')):
            continue
        img = cv2.imread(os.path.join(IMAGE_DIR, path))
        # img = cv2.resize(img, dsize=(1024,1024), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
        predictor.set_image(img)
        # seg_map = load_seg_label(os.path.join(IMAGE_DIR, path), args.semantic_gt_path)
        coords, seg_map = load_depth(os.path.join(IMAGE_DIR, path), args.semantic_gt_path)
        seg_map = seg_map + 1
        num_labels = np.unique(seg_map)
        mask_list = []
        for label in num_labels:
            index = seg_map==label
            coord_label = coords[index]
            seg_base = np.zeros_like(seg_map)
            seg_base[index] = label
            seg_map_label = seg_map[index]

            masks, scores, _ = predictor.predict(
                point_coords=coord_label,
                point_labels=seg_map_label,
                multimask_output=True,
            )
            mask_inx = np.where(scores==np.max(scores))
            mask = masks[mask_inx]
        #     mask_list.append(mask[0])
        # masks = torch.tensor(np.stack(mask_list, 0))
        # torch.save(masks, os.path.join(OUTPUT_DIR, name+'.pt'))
            print(mask.shape) #output: (1, 600, 900)

            plt.figure(figsize=(10,10))
            plt.imshow(img)
            # show_mask(mask, plt.gca(),random_color=False)
            show_points(coord_label, seg_map_label, plt.gca())
            plt.axis('off')
            plt.savefig(os.path.join(OUTPUT_DIR, str(label) + '_' + name +'.png'))
            plt.close()
