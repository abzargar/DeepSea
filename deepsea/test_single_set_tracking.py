import os
import tracker_transforms
import numpy as np
import argparse
import random
import cv2
import torch
from model import DeepSeaTracker
from test_single_image_segmentation import apply_img_segmentation
from utils import track_cells

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def apply_cell_tracking(seg_model_ckpt,tr_model_ckpt,img_list,tr_image_size = [128,128],image_means = [0.5],image_stds= [0.5]):
    """ function to cell tracking process to a single set of time lapse microscopy image sequences

            Parameters
            ------------

            seg_model_ckpt: Pre-trained segmentation model checkpoint
            tr_model_ckpt: Pre-trained tracking model checkpoint
            img: Input image (PIL Image)

            Returns
            ------------

            cell_labels: list of frame cell labels
            cell_centroids: list of frame cell centroids
            tracked_imgs: list of tracked images

        """
    print('Run segmentation process ...')
    label_img_list,new_img_list=[],[]
    for img in img_list:
        label_img,_,_,img_resized=apply_img_segmentation(seg_model_ckpt,img)
        label_img_list.append(label_img)
        new_img_list.append(img_resized)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tr_transforms = tracker_transforms.Compose([
        tracker_transforms.Resize(tr_image_size),
        tracker_transforms.Grayscale(num_output_channels=1),
        tracker_transforms.ToTensor(),
        tracker_transforms.Normalize(mean=image_means,
                                     std=image_stds)
    ])

    tracking_model = DeepSeaTracker(n_channels=1, n_classes=2, bilinear=True)
    tracking_model.load_state_dict(tr_model_ckpt)
    tracking_model = tracking_model.to(device)
    print('Run cell tracking process ...')
    cell_labels,cell_centroids,tracked_imgs=track_cells(label_img_list,new_img_list,tracking_model,device,transforms=tr_transforms)
    return cell_labels,cell_centroids,tracked_imgs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--single_image_set_dir",required=True,type=str,help="path for the single time lapse microscopy image sequences")
    ap.add_argument("--seg_ckpt_dir",required=True,type=str,help="path for the checkpoint of segmentation model")
    ap.add_argument("--tr_ckpt_dir",required=True,type=str,help="path for the checkpoint of tracking model")
    ap.add_argument("--output_dir", required=True, type=str, help="path for saving the test outputs")

    args = ap.parse_args()

    assert os.path.isdir(args.single_image_set_dir), 'No such file or directory: ' + args.single_image_set_dir
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)


    seg_model_ckpt = torch.load(args.seg_ckpt_dir)
    tr_model_ckpt = torch.load(args.tr_ckpt_dir)
    print('Read images ...')
    img_list=[]
    for img_name in sorted(os.listdir(args.single_image_set_dir)):
        img_list.append(cv2.imread(os.path.join(args.single_image_set_dir,img_name),0))

    cell_labels,cell_centroids,tracked_imgs=apply_cell_tracking(seg_model_ckpt,tr_model_ckpt,img_list)
    if tracked_imgs:
        for id, img in enumerate(tracked_imgs):
            cv2.imwrite(os.path.join(args.output_dir, 'img_{:04d}.png'.format(id)), img)