import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from skimage.morphology import remove_small_objects
import copy
import cv2
import os
from loss import multiclass_dice_coeff
from scipy.optimize import linear_sum_assignment
from utils import visualize_segmentation
from scipy import ndimage as ndi


def evaluate_segmentation(net, valid_iterator, device,n_valid_examples,is_avg_prec=False,prec_thresholds=[0.5,0.7,0.9],output_dir=None):
    net.eval()
    num_val_batches = len(valid_iterator)
    dice_score = 0
    mask_list, pred_list,wmap_list = [], [],[]
    # iterate over the validation set
    # loss=0
    with tqdm(total=n_valid_examples, desc='Segmentation Val round', unit='img') as pbar:
        for batch_idx,batch in enumerate(valid_iterator):
            images, true_masks,wmap = batch['image'], batch['mask'],batch['wmap']
            images_device = images.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.long)
            true_masks = torch.squeeze(true_masks, dim=1)
            true_masks_copy = copy.deepcopy(true_masks)
            true_masks = F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float()

            with torch.no_grad():
                # predict the mask
                mask_pred,edge_pred = net(images_device)

                # convert to one-hot format
                mask_pred_copy = copy.deepcopy(mask_pred.argmax(dim=1))
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], true_masks[:, 1:, ...],
                                                    reduce_batch_first=False)
                # loss += dice_loss(F.softmax(mask_pred, dim=1).float(),
                #                   F.one_hot(true_masks_copy, net.n_classes).permute(0, 3, 1, 2).float(),
                #                   multiclass=True)
                if is_avg_prec:
                    true_masks_copy=true_masks_copy.cpu().numpy()
                    mask_pred_copy = mask_pred_copy.cpu().numpy()
                    for i in range(true_masks_copy.shape[0]):
                        mask,_=ndi.label(remove_small_objects(true_masks_copy[i,:,:]>0,min_size=20,connectivity=1))
                        mask_list.append(mask)
                        pred, _ = ndi.label(remove_small_objects(mask_pred_copy[i, :, :]>0,min_size=20,connectivity=1))
                        if output_dir:
                            img=images[i].cpu().numpy()[0, :, :]
                            img=(img-np.min(img))/(np.max(img)-np.min(img))*255
                            overlay_img = visualize_segmentation(pred, inp_img=img, overlay_img=True)
                            cv2.imwrite(os.path.join(output_dir,'input_segmentation_images','images_{:04d}.png'.format(batch_idx*true_masks_copy.shape[0]+i)),img)
                            cv2.imwrite(os.path.join(output_dir, 'segmentation_predictions', 'images_{:04d}.png'.format(batch_idx*true_masks_copy.shape[0]+ i)),overlay_img)

                        wmap_list.append(wmap[i].cpu().numpy()[0, :, :])
                        pred_list.append(pred)

            pbar.update(images.shape[0])


    if is_avg_prec:
        avg_list=average_precision(mask_list, pred_list, threshold=prec_thresholds)[0]
        easy_samples,hard_samples=[],[]
        for i,wmap in enumerate(wmap_list):
            if np.sum(wmap)==0:
                easy_samples.append(avg_list[i])
            else:
                hard_samples.append(avg_list[i])
        if output_dir:
            np.savetxt(os.path.join(output_dir,'precisions.txt'), avg_list, delimiter=',')
        avg_prec=np.mean(avg_list,axis=0)
        easy_avg_prec=np.mean(np.array(easy_samples),axis=0)
        hard_avg_prec = np.mean(np.array(hard_samples), axis=0)
        return dice_score.cpu().numpy() / num_val_batches, avg_prec, easy_avg_prec, hard_avg_prec

    return dice_score.cpu().numpy() / num_val_batches, None,None,None


def evaluate_tracker(net, valid_iterator, device,n_valid_examples,is_avg_prec=False,prec_thresholds=[0.5,0.7,0.9],output_dir=None):
    net.eval()
    num_val_batches = len(valid_iterator)
    dice_score = 0
    mask_list, pred_list= [], []
    # iterate over the validation set
    with tqdm(total=n_valid_examples, desc='Tracking Val round', unit='img') as pbar:
        for batch_idx,batch in enumerate(valid_iterator):
            images_prev, images_curr,true_masks = batch['image_prev'], batch['image_curr'],batch['mask']

            images_device_prev = images_prev.to(device=device, dtype=torch.float32)
            images_device_curr = images_curr.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.long)
            true_masks = torch.squeeze(true_masks, dim=1)
            true_masks_copy = copy.deepcopy(true_masks)
            true_masks = F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float()

            with torch.no_grad():
                # predict the mask
                mask_pred = net(images_device_prev,images_device_curr)

                # convert to one-hot format
                mask_pred_copy = copy.deepcopy(mask_pred.argmax(dim=1))
                # edge_pred_copy = copy.deepcopy(edge_pred.argmax(dim=1))
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], true_masks[:, 1:, ...],
                                                        reduce_batch_first=False)
                if is_avg_prec:
                    true_masks_copy=true_masks_copy.cpu().numpy()
                    mask_pred_copy = mask_pred_copy.cpu().numpy()
                    for i in range(true_masks_copy.shape[0]):
                        mask,_=ndi.label(remove_small_objects(true_masks_copy[i,:,:]>0,min_size=20,connectivity=1))
                        mask_list.append(mask)
                        pred, _ = ndi.label(remove_small_objects(mask_pred_copy[i, :, :]>0,min_size=20,connectivity=1))

                        if output_dir:
                            img=images_prev[i].cpu().numpy()[0, :, :]
                            img=(img-np.min(img))/(np.max(img)-np.min(img))*255
                            overlay_img = visualize_segmentation(pred, inp_img=img, overlay_img=True)
                            cv2.imwrite(os.path.join(output_dir,'input_tracking_images','images_{:04d}.png'.format(batch_idx*true_masks_copy.shape[0]+i)),img)
                            cv2.imwrite(os.path.join(output_dir, 'tracking_predictions', 'images_{:04d}.png'.format(batch_idx*true_masks_copy.shape[0]+ i)),overlay_img)

                        pred_list.append(pred)

            pbar.update(images_prev.shape[0])

    if is_avg_prec:
        avg_list = average_precision(mask_list, pred_list, threshold=prec_thresholds)[0]
        single_cell_tracking, mitosis_tracking = [], []
        for i, mask in enumerate(mask_list):
            if np.max(mask) >1:
                mitosis_tracking.append(avg_list[i])
            else:
                single_cell_tracking.append(avg_list[i])

        if output_dir:
            np.savetxt(os.path.join(output_dir, 'precisions.txt'), avg_list, delimiter=',')
        avg_prec = np.mean(avg_list, axis=0)
        single_cell_avg_prec = np.mean(np.array(single_cell_tracking), axis=0)
        mitosis_avg_prec = np.mean(np.array(mitosis_tracking), axis=0)
        return dice_score.cpu().numpy() / num_val_batches, avg_prec, single_cell_avg_prec, mitosis_avg_prec

    return dice_score.cpu().numpy() / num_val_batches, None, None, None

def _label_overlap(x, y):
    """ fast function to get pixel overlaps between masks in x and y

    Parameters
    ------------

    x: ND-array, int
        where 0=NO masks; 1,2... are mask labels
    y: ND-array, int
        where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    overlap: ND-array, int
        matrix of pixel overlaps of size [x.max()+1, y.max()+1]

    """
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint)
    for i in range(len(x)):
        overlap[x[i], y[i]] += 1
    return overlap

def _intersection_over_union(masks_true, masks_pred):
    """ intersection over union of all mask pairs

    Parameters
    ------------

    masks_true: ND-array, int
        ground truth masks, where 0=NO masks; 1,2... are mask labels
    masks_pred: ND-array, int
        predicted masks, where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    iou: ND-array, float
        matrix of IOU pairs of size [x.max()+1, y.max()+1]

    """
    overlap = _label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap+1e-6)
    iou[np.isnan(iou)] = 0.0
    return iou


def _true_positive(iou, th):
    """ true positive at threshold th

    Parameters
    ------------

    iou: float, ND-array
        array of IOU pairs
    th: float
        threshold on IOU for positive label

    Returns
    ------------

    tp: float
        number of true positives at threshold

    """
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= th).astype(float) - iou / (2 * n_min+1e-6)
    true_ind, pred_ind = linear_sum_assignment(costs)
    match_ok = iou[true_ind, pred_ind] >= th
    tp = match_ok.sum()
    return tp



def average_precision(masks_true, masks_pred, threshold=[0.5, 0.75, 0.9]):
    """ average precision estimation: AP = TP / (TP + FP + FN)

    This function is based heavily on the *fast* stardist matching functions
    (https://github.com/mpicbg-csbd/stardist/blob/master/stardist/matching.py)

    Parameters
    ------------

    masks_true: list of ND-arrays (int) or ND-array (int)
        where 0=NO masks; 1,2... are mask labels
    masks_pred: list of ND-arrays (int) or ND-array (int)
        ND-array (int) where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    ap: array [len(masks_true) x len(threshold)]
        average precision at thresholds
    tp: array [len(masks_true) x len(threshold)]
        number of true positives at thresholds
    fp: array [len(masks_true) x len(threshold)]
        number of false positives at thresholds
    fn: array [len(masks_true) x len(threshold)]
        number of false negatives at thresholds

    """
    not_list = False
    if not isinstance(masks_true, list):
        masks_true = [masks_true]
        masks_pred = [masks_pred]
        not_list = True
    if not isinstance(threshold, list) and not isinstance(threshold, np.ndarray):
        threshold = [threshold]
    ap = np.zeros((len(masks_true), len(threshold)), np.float32)
    tp = np.zeros((len(masks_true), len(threshold)), np.float32)
    fp = np.zeros((len(masks_true), len(threshold)), np.float32)
    fn = np.zeros((len(masks_true), len(threshold)), np.float32)
    n_true = np.array(list(map(np.max, masks_true)))
    n_pred = np.array(list(map(np.max, masks_pred)))
    with tqdm(total=len(masks_true), desc='Precision measurement', unit='img') as pbar:
        for n in range(len(masks_true)):
            if n_pred[n] > 0:
                iou = _intersection_over_union(masks_true[n], masks_pred[n])[1:, 1:]
                for k, th in enumerate(threshold):
                    tp[n, k] = _true_positive(iou, th)
            fp[n] = n_pred[n] - tp[n]
            fn[n] = n_true[n] - tp[n]
            ap[n] = tp[n] / (tp[n] + fp[n] + fn[n]+1e-6)
            pbar.update(1)
    if not_list:
        ap, tp, fp, fn = ap[0], tp[0], fp[0], fn[0]
    return ap, tp, fp, fn