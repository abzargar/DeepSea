import torch.utils.data as data
from PIL import Image
import collections
import shutil
from scipy.spatial import distance
import tracker_transforms
import segmentation_transforms
from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects
import numpy as np
import cv2
import argparse
from tqdm import tqdm
import os
import random
from model import DeepSeaTracker,DeepSeaSegmentation
from data import BasicSegmentationDataset
import torch
from utils import add_labels_to_image,read_labeles,\
    remove_detected_objects,evaluate_tracking_results

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def preprocess(pil_img_prev, pil_curr, pil_mask, transforms):
    tensor_img_prev, tensor_img_curr, tensor_mask = transforms(pil_img_prev, pil_curr, pil_mask)
    return tensor_img_prev, tensor_img_curr, tensor_mask

def find_matchings(model_tracker, img_curr, img_prev, masks_curr, masks_prev, mask_curr, centroids_curr,
                   centroids_prev, cell_labels, last_new_label_index,device,transforms):
    labels_curr = []
    centroids_curr_new = []
    masks_curr_new = []
    scores = []
    labels_prev=cell_labels[-1]

    mask_curr = mask_curr.astype('float64')
    argmin_list = []
    for idx, mask_prev in enumerate(masks_prev):
        centroid = np.array(ndi.measurements.center_of_mass(mask_prev>0))
        tmp = np.where(mask_prev > 0)
        window_size = np.max([(np.max(tmp[0]) - np.min(tmp[0])) * 6, (np.max(tmp[1]) - np.min(tmp[1])) * 6])
        ROI_mask_prev = Image.fromarray((mask_prev > 0).astype('float32') * img_prev)
        ROI_mask_prev = ROI_mask_prev.crop((int(centroid[1] - window_size / 2), int(centroid[0] - window_size / 2),
                                    int(centroid[1] + window_size / 2), int(centroid[0] + window_size / 2)))

        ROI_mask_curr = Image.fromarray((mask_curr > 0).astype('float64') * img_curr)
        ROI_mask_curr=ROI_mask_curr.crop((int(centroid[1] - window_size / 2), int(centroid[0] - window_size / 2),
                            int(centroid[1] + window_size / 2), int(centroid[0] + window_size / 2)))

        tensor_img_prev, tensor_img_curr, _ = preprocess(ROI_mask_prev, ROI_mask_curr, ROI_mask_curr, transforms)

        tensor_img_prev = tensor_img_prev.to(device=device, dtype=torch.float32)
        tensor_img_curr = tensor_img_curr.to(device=device, dtype=torch.float32)
        mask_pred = model_tracker(torch.unsqueeze(tensor_img_prev, 0), torch.unsqueeze(tensor_img_curr, 0))
        mask_pred = mask_pred.argmax(dim=1)
        mask_pred=mask_pred.cpu().numpy()
        mask_pred=remove_small_objects(mask_pred[0, :, :] > 0, min_size=20, connectivity=1).astype('float64')
        mask_curr_cropped = Image.fromarray((mask_curr).astype('float64'))
        mask_curr_cropped = mask_curr_cropped.crop((int(centroid[1] - window_size / 2), int(centroid[0] - window_size / 2),
                                            int(centroid[1] + window_size / 2), int(centroid[0] + window_size / 2)))
        mask_curr_cropped=np.asarray(mask_curr_cropped).astype('float32')
        labels_idx = np.unique(np.round(mask_curr_cropped))
        labels_idx = labels_idx[labels_idx != 0]

        if np.sum(mask_curr_cropped):
            centroid_list,iou_list,dst_min_list=[],[],[]
            for j in labels_idx:
                markers = cv2.resize((mask_curr_cropped == j).astype('float64'), (128, 128),cv2.INTER_NEAREST).astype('int')
                iou = np.sum(markers * mask_pred) / (np.sum(markers)+1e-6)
                if iou > 0.5:
                    centroid = np.array(ndi.measurements.center_of_mass(mask_curr==j))
                    centroid_list.append(np.round(centroid).astype('int').tolist())
                    iou_list.append(iou)
                    dst_list = np.array(
                        [distance.euclidean(centroid, point) for point in centroids_curr])
                    dst_min_list.append(np.argmin(dst_list))
            if len(centroid_list)==1:
                labels_curr.append(labels_prev[idx])
                centroids_curr_new.append(centroid_list[0])
                scores.append(iou_list[0])
                masks_curr_new.append(masks_curr[dst_min_list[0]])
                argmin_list.append(dst_min_list[0])
            elif len(centroid_list)>1:
                iou_list_sorted_index=np.argsort(np.array(iou_list))
                err = 0
                if '_' in labels_prev[idx]:
                    for labels in cell_labels[-5:-1]:
                        if '_'.join(labels_prev[idx].split('_')[:-1]) in labels:
                            err = 1
                            break
                if err == 0:
                    for i in range(2):
                            labels_curr.append(labels_prev[idx]+'_'+str(i))
                            centroids_curr_new.append(centroid_list[iou_list_sorted_index[-i-1]])
                            scores.append(iou_list[iou_list_sorted_index[-i-1]])
                            masks_curr_new.append(masks_curr[dst_min_list[iou_list_sorted_index[-i-1]]])
                            argmin_list.append(dst_min_list[iou_list_sorted_index[-i-1]])
                else:
                    centroid=centroids_prev[labels_prev.index(labels_prev[idx])]
                    centroid_1=centroid_list[iou_list_sorted_index[-1]]
                    centroid_2 = centroid_list[iou_list_sorted_index[-2]]
                    dst1 = distance.euclidean(centroid, centroid_1)
                    dst2 = distance.euclidean(centroid, centroid_2)
                    if dst1 < dst2:
                        labels_curr.append(labels_prev[idx])
                        centroids_curr_new.append(centroid_list[iou_list_sorted_index[-1]])
                        scores.append(iou_list[iou_list_sorted_index[-1]])
                        masks_curr_new.append(masks_curr[dst_min_list[iou_list_sorted_index[-1]]])
                        argmin_list.append(dst_min_list[iou_list_sorted_index[-1]])
                    else:
                        labels_curr.append(labels_prev[idx])
                        centroids_curr_new.append(centroid_list[iou_list_sorted_index[-2]])
                        scores.append(iou_list[iou_list_sorted_index[-2]])
                        masks_curr_new.append(masks_curr[dst_min_list[iou_list_sorted_index[-2]]])
                        argmin_list.append(dst_min_list[iou_list_sorted_index[-2]])


    for idx in range(len(centroids_curr)):
        if idx not in argmin_list:
            last_new_label_index = last_new_label_index + 1
            labels_curr.append(str(last_new_label_index))
            centroids_curr_new.append(centroids_curr[idx])
            masks_curr_new.append(masks_curr[idx])

    duplicates_centroids = collections.defaultdict(list)
    for idx, centroid in enumerate(centroids_curr_new):
        if centroids_curr_new.count(centroid) > 1:
            duplicates_centroids[(centroid[0], centroid[1])].append(idx)

    if duplicates_centroids:
        remove_list = []
        for centroid in duplicates_centroids:
            label_1 = labels_curr[duplicates_centroids[centroid][0]]
            label_2 = labels_curr[duplicates_centroids[centroid][1]]
            if label_1 not in labels_prev:
                label_1='_'.join(label_1.split('_')[:-1])
            if label_2 not in labels_prev:
                label_2='_'.join(label_2.split('_')[:-1])
            centroid_1 = centroids_prev[labels_prev.index(label_1)]
            centroid_2 = centroids_prev[labels_prev.index(label_2)]
            dst1 = distance.euclidean(centroid, centroid_1)
            dst2 = distance.euclidean(centroid, centroid_2)
            if dst1 < dst2:
                remove_list.append(duplicates_centroids[centroid][1])
            else:
                remove_list.append(duplicates_centroids[centroid][0])

        if remove_list:
            centroids_curr_new, masks_curr_new, labels_curr = remove_detected_objects(remove_list,
                                                                                      centroids_curr_new,
                                                                                      masks_curr_new, labels_curr)

    duplicates_labels = collections.defaultdict(list)
    duplicates_labels_scores = collections.defaultdict(list)
    for idx, label in enumerate(labels_curr):
        if labels_curr.count(label) > 1:
            duplicates_labels[label].append(idx)
            duplicates_labels_scores[label].append(scores[idx])

    if duplicates_labels:
        remove_list = []
        for label_idx, label in enumerate(duplicates_labels):
            flag = 0
            for idx, center in enumerate(centroids_curr_new):
                if idx in duplicates_labels[label]:
                    continue
                for j in range(len(duplicates_labels[label])):
                    dst = distance.euclidean(centroids_curr_new[duplicates_labels[label][j]], center)

                    if dst < 5:
                        flag = 1
                        if scores[duplicates_labels[label][j]] < scores[idx]:
                            remove_list.append(duplicates_labels[label][j])

            if flag == 0:
                labels_curr[duplicates_labels[label][0]] = label + '_1'
                labels_curr[duplicates_labels[label][1]] = label + '_2'

        if remove_list:
            centroids_curr_new, masks_curr_new, labels_curr = remove_detected_objects(remove_list,
                                                                                      centroids_curr_new,
                                                                                      masks_curr_new, labels_curr)

    return labels_curr, masks_curr_new, centroids_curr_new, last_new_label_index

def track_cells(pred_list,img_list,tracker_model,device,transforms,output_dir=None):
    tracker_model.eval()
    cell_labels,cell_centroids=[],[]
    with tqdm(total=len(pred_list), desc='Tracker Val round', unit='img') as pbar:
        for pred_id, (mask_curr,img_curr) in enumerate(zip(pred_list,img_list)):
            centroids_curr, masks_curr = [], []
            for label_id in range(1, np.max(mask_curr) + 1):
                mask = mask_curr == label_id
                centroid = np.array(ndi.measurements.center_of_mass(mask))
                cell_centroid=np.round(centroid).astype('int').tolist()
                centroids_curr.append([cell_centroid[0], cell_centroid[1]])
                masks_curr.append(mask)

            if pred_id == 0:
                labels_curr = [str(i) for i in range(len(centroids_curr))]
                last_new_label_index = len(centroids_curr) - 1
            else:
                labels_curr, masks_curr, centroids_curr, last_new_label_index = find_matchings(tracker_model, img_curr,
                                                                                               img_prev, masks_curr,
                                                                                               masks_prev, mask_curr,
                                                                                               centroids_curr,
                                                                                               centroids_prev,
                                                                                               cell_labels,
                                                                                               last_new_label_index,device,transforms)
            if output_dir is not None:
                cv2.imwrite(os.path.join(output_dir,'tracked_images', 'img_'+str(pred_id)+'.png'), add_labels_to_image(img_curr,labels_curr,centroids_curr))
            cell_labels.append(labels_curr)
            cell_centroids.append((np.array(centroids_curr)).tolist())
            centroids_prev = centroids_curr
            img_prev = img_curr
            masks_prev = masks_curr
            pbar.update(1)

    return cell_labels,cell_centroids

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
                               segmentation_transforms.Resize(seg_img_size),
                               segmentation_transforms.Grayscale(num_output_channels=1),
                               segmentation_transforms.ToTensor(),
                               segmentation_transforms.Normalize(mean = image_means,
                                                    std = image_stds)
                           ])

    test_data = BasicSegmentationDataset(os.path.join(args.test_dir, 'images'), os.path.join(args.test_dir, 'masks'),
                             os.path.join(args.test_dir, 'wmaps'), transforms=seg_transforms,mask_suffix= '_cell_area_masked')

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

    tracker_model = DeepSeaTracker(n_channels=1, n_classes=2, bilinear=True)
    tracker_model.load_state_dict(torch.load(args.tracker_ckpt_dir))
    tracker_model = tracker_model.to(device)

    gt_cell_labels, gt_cell_centroids = read_labeles(args.test_dir, seg_img_size)
    cell_labels,cell_centroids=track_cells(pred_list,img_list,tracker_model,device,transforms=tr_transforms,output_dir=args.output_dir)

    results = evaluate_tracking_results(cell_labels, cell_centroids, gt_cell_labels, gt_cell_centroids)
    print('Results:',results)



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir",required=True,type=str,help="path for the time-lapse microscopy frame sequences")
    ap.add_argument("--seg_ckpt_dir",required=True,type=str,help="path for the checkpoint of segmentation model")
    ap.add_argument("--tracker_ckpt_dir", required=True, type=str, help="path for the checkpoint of tracker model")
    ap.add_argument("--output_dir", required=True, type=str, help="path for saving the test outputs")

    args = ap.parse_args()

    assert os.path.isdir(args.test_dir), 'No such file or directory: ' + args.test_dir

    if os.path.isdir(args.output_dir+'/tracked_images'):
        shutil.rmtree(args.output_dir+'/tracked_images')
    os.makedirs(args.output_dir+'/tracked_images')
    main(args)





