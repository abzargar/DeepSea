import torch.utils.data as data
import shutil
import tracker_transforms
import segmentation_transforms
from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects
import numpy as np
from utils import track_cells
import argparse
from tqdm import tqdm
import cv2
import os
import random
from model import DeepSeaTracker,DeepSeaSegmentation
from data import BasicSegmentationDataset
import torch
from utils import read_labeles,evaluate_tracking_results

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True



def get_segmented_images(net, test_iterator, device,n_examples):
    net.eval()
    pred_list= []
    img_list=[]
    with tqdm(total=n_examples, desc='Segmentation Val round', unit='img') as pbar:
        for batch_idx,batch in enumerate(test_iterator):
            images, true_masks,wmap = batch['image'], batch['mask'],batch['wmap']
            images_device = images.to(device=device, dtype=torch.float32)
            with torch.no_grad():
                # predict the mask
                mask_pred,edge_pred = net(images_device)
                mask_pred=mask_pred.argmax(dim=1)
                mask_pred = mask_pred.cpu().numpy()
                for i in range(images.shape[0]):
                    pred, _ = ndi.label(remove_small_objects(mask_pred[i, :, :]>0,min_size=20,connectivity=1))
                    img = images[i].cpu().numpy()[i, :, :]
                    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
                    pred_list.append(pred)
                    img_list.append(img)
            pbar.update(images.shape[0])
    return pred_list,img_list

def main(args,seg_img_size= [383,512],tracking_image_size = [128,128],image_means = [0.5],image_stds= [0.5],batch_size=1):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seg_transforms = segmentation_transforms.Compose([
                               segmentation_transforms.ToPILImage(),
                               segmentation_transforms.Resize(seg_img_size),
                               segmentation_transforms.ToTensor(),
                               segmentation_transforms.Normalize(mean = image_means,
                                                    std = image_stds)
                           ])

    test_data = BasicSegmentationDataset(os.path.join(args.single_test_set_dir, 'images'), os.path.join(args.single_test_set_dir, 'masks'),
                             os.path.join(args.single_test_set_dir, 'wmaps'), transforms=seg_transforms,mask_suffix= '_cell_area_masked')

    test_iterator = data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    seg_model = DeepSeaSegmentation(n_channels=1, n_classes=2, bilinear=True)

    seg_model.load_state_dict(torch.load(args.seg_ckpt_dir))
    seg_model = seg_model.to(device)

    pred_list,img_list=get_segmented_images(seg_model, test_iterator, device,len(test_data))

    tr_transforms = tracker_transforms.Compose([
        tracker_transforms.Resize(tracking_image_size),
        tracker_transforms.Grayscale(num_output_channels=1),
        tracker_transforms.ToTensor(),
        tracker_transforms.Normalize(mean=image_means,
                             std=image_stds)
    ])

    tracking_model = DeepSeaTracker(n_channels=1, n_classes=2, bilinear=True)
    tracking_model.load_state_dict(torch.load(args.tr_ckpt_dir))
    tracking_model = tracking_model.to(device)

    gt_cell_labels, gt_cell_centroids = read_labeles(args.single_test_set_dir, seg_img_size)
    cell_labels,cell_centroids,tracked_imgs=track_cells(pred_list,img_list,tracking_model,device,transforms=tr_transforms)
    if args.output_dir is not None:
        for id,img in enumerate(tracked_imgs):
            cv2.imwrite(os.path.join(args.output_dir, 'img_{:04d}.png'.format(id)),img)
    results = evaluate_tracking_results(cell_labels, cell_centroids, gt_cell_labels, gt_cell_centroids)
    print('Results:',results)



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--single_test_set_dir",required=True,type=str,help="path for the time-lapse microscopy test set")
    ap.add_argument("--seg_ckpt_dir",required=True,type=str,help="path for the checkpoint of segmentation model")
    ap.add_argument("--tr_ckpt_dir", required=True, type=str, help="path for the checkpoint of tracking model")
    ap.add_argument("--output_dir", required=True, type=str, help="path for saving the test outputs")

    args = ap.parse_args()
    assert os.path.isdir(args.single_test_set_dir), 'No such file or directory: ' + args.single_test_set_dir

    if os.path.isdir(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)
    main(args)


