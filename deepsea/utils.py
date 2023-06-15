from numpy import unique,ones,minimum
from scipy.ndimage import minimum_filter
from skimage.segmentation import watershed
import random
import cv2
import os
import torch
from tqdm import tqdm
import collections
from PIL import Image
from scipy.spatial import distance
from munkres import Munkres
from collections import defaultdict
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects,reconstruction

DATA_LOADER_SEED = 9
random.seed(DATA_LOADER_SEED)
class_colors = [(0,0,0)]+[(random.randint(50, 255), random.randint(
    50, 255), random.randint(50, 255)) for _ in range(1000)]


def evaluate_tracking_results(cell_labels, cell_centroids, gt_cell_labels, gt_cell_centroids):
    seq_trajectories = defaultdict(list)
    gt_tracker = []
    gt_id_switch = []
    gt_fragmentation = []
    n_gt, n_tr = 0, 0
    max_cost = 1e9
    hm = Munkres()
    tp, fn, fp = 0, 0, 0

    for idx, (cell_label, cell_centroid, gt_cell_label, gt_cell_centroid) in enumerate(
            zip(cell_labels, cell_centroids, gt_cell_labels, gt_cell_centroids)):

        gt_cell_label, gt_cell_centroid = remove_objects_from_list(gt_cell_label, gt_cell_centroid, pad_size=80)

        cell_label, cell_centroid = remove_objects_from_list(cell_label, cell_centroid, pad_size=80)

        n_tr += len(cell_label)
        n_gt += len(gt_cell_label)

        cost_matrix = []
        frame_ids = [[], []]
        # loop over ground truth objects in one frame
        for idx, (gt_label, gt_centroid) in enumerate(zip(gt_cell_label, gt_cell_centroid)):
            frame_ids[0].append(gt_label)
            frame_ids[1].append(-1)
            gt_tracker.append(-1)
            gt_id_switch.append(0)
            gt_fragmentation.append(0)
            cost_row = []
            # loop over tracked objects in one frame
            for label, centroid in zip(cell_label, cell_centroid):
                dst = distance.euclidean(centroid, gt_centroid)
                if dst >= 6:
                    cost_row.append(max_cost)
                else:
                    cost_row.append(1)

            cost_matrix.append(cost_row)
            seq_trajectories[gt_label].append(-1)
        association_matrix = hm.compute(cost_matrix)
        tmptp = 0
        # mapping for tracker ids and ground truth ids
        for row, col in association_matrix:
            c = cost_matrix[row][col]
            if c < max_cost:
                seq_trajectories[gt_cell_label[row]][-1] = cell_label[col]

                # true positives are only valid associations
                tp += 1
                tmptp += 1

        fn += len(gt_cell_label) - len(association_matrix)
        fp += len(cell_label) - tmptp
    mostly_lost, mostly_tracked, id_switches, fragments = 0, 0, 0, 0
    for g in seq_trajectories.values():
        # all frames of this gt trajectory are not assigned to any detections
        if all([this == -1 for this in g]):
            mostly_lost += 1
            continue
        # compute tracked frames in trajectory
        last_id = g[0]
        tracked = 1 if g[0] != -1 else 0
        for f in range(1, len(g)):
            if last_id != g[f] and last_id != -1 and g[f] != -1 and g[f - 1] != -1:
                id_switches += 1
            if f < len(g) - 1 and g[f - 1] != g[f] and last_id != -1 and g[f] != -1 and g[f + 1] != -1:
                fragments += 1
            if g[f] != -1:
                tracked += 1
                last_id = g[f]
        # handle last frame; tracked state is handled in for loop (g[f]!=-1)
        if len(g) > 1 and g[f - 1] != g[f] and last_id != -1 and g[f] != -1:
            fragments += 1
        # compute mostly_tracked/partialy_tracked/mostly_lost
        tracking_ratio = tracked / len(g)
        if tracking_ratio > 0.8:
            mostly_tracked += 1
        elif tracking_ratio < 0.2:
            mostly_lost += 1

    number_of_trajectories = len(seq_trajectories)
    mostly_tracked /= number_of_trajectories
    mostly_lost /= number_of_trajectories

    if (fp + tp) == 0 or (tp + fn) == 0:
        recall = 0.
        precision = 0.
    else:
        recall = tp / (tp + fn)
        precision = tp / (fp + tp)

    MOTA = 1 - (fp + fn + id_switches) / n_gt

    results = {
        "MOTA": MOTA,
        "Number of trajectories": number_of_trajectories,
        "MT": mostly_tracked,
        "ML": mostly_lost,
        "Precision": precision,
        "Recall": recall,
        "Frag": fragments,
        "IDS": id_switches
    }
    return results

def remove_detected_objects(remove_list,centroids,masks,labels):
    temp_list_1,temp_list_2,temp_list_3=[],[],[]
    for idx in range(len(labels)):
        if idx in remove_list:
            continue
        temp_list_1.append(centroids[idx])
        temp_list_2.append(masks[idx])
        temp_list_3.append(labels[idx])
    centroids=temp_list_1
    masks=temp_list_2
    labels=temp_list_3
    return centroids,masks,labels

def remove_objects_from_list(cell_label, cell_centroid,pad_size=15):
    centroid_remove_list, label_remove_list = [], []
    for label, centroid in zip(cell_label, cell_centroid):
        if centroid[0] < pad_size or centroid[0] > (512 - pad_size) or centroid[1] < pad_size or centroid[1] > (
                512 - pad_size):
            centroid_remove_list.append(centroid)
            label_remove_list.append(label)
    for label, centroid in zip(label_remove_list, centroid_remove_list):
        cell_centroid.remove(centroid)
        cell_label.remove(label)
    return cell_label, cell_centroid

def add_labels_to_image(img, labels, centroids, fontScale=0.4, thickness=1, color=(100, 255, 50)):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for idx, label in enumerate(labels):
        point = [x + y for x, y in zip(centroids[idx][::-1], [-5, 3])]
        img = cv2.putText(img, label, tuple(point), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness);
    return img

def read_labeles(test_path,seg_img_size):
    ACCEPTABLE_IMAGE_FORMATS = [".txt"]
    cell_labels = []
    cell_centroids = []
    labels_path=os.path.join(test_path,'labels')
    images_path=os.path.join(test_path, 'images')
    label_file_list=sorted(os.listdir(labels_path))
    images_file_list = sorted(os.listdir(images_path))
    for label_dir_entry,img_dir_entry in zip(label_file_list,images_file_list):
        if os.path.isfile(os.path.join(labels_path, label_dir_entry)) and \
                os.path.splitext(label_dir_entry)[1] in ACCEPTABLE_IMAGE_FORMATS:
            file = open(os.path.join(labels_path, label_dir_entry), "r")
            lines = file.readlines()
            lines.pop(0)
            img=cv2.imread(os.path.join(images_path, img_dir_entry))
            labels, centroids = [], []
            for line in lines:
                labels.append(line.replace('\n', '').split('\t')[0])
                centroid_0=float(line.replace('\n', '').split('\t')[1])
                centroid_1 = float(line.replace('\n', '').split('\t')[2])
                centroids.append([np.round((centroid_1/img.shape[0]*seg_img_size[0])),np.round((centroid_0/ img.shape[1] * seg_img_size[1]))])

            cell_labels.append(labels)
            cell_centroids.append(centroids)

    return cell_labels, cell_centroids

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def imextendedmin(im, thresh,conn=3):
    """Suppress all minima that are shallower than thresh.
    Parameters
    ----------
    a : array
        The input array on which to perform hminima.
    thresh : float
        Any local minima shallower than this will be flattened.
    Returns
    -------
    out : array
        A copy of the input array with shallow minima suppressed.
    """
    # maxval = a.max()
    # ainv = maxval-a
    im=complement(im)
    return regional_minima(-reconstruction(im-thresh, im),conn)

def imimposemin(im, BW, conn=3):
    """Transform 'a' so that its only regional minima are those in 'minima'.

    Parameters:
        'a': an ndarray
        'minima': a boolean array of same shape as 'a'
        'connectivity': the connectivity of the structuring element used in
        morphological reconstruction.
    Value:
        an ndarray of same shape as a with unmarked local minima paved over.
    """
    fm = im.copy()
    fm[BW] = -1e4

    fm[~BW] = 1e4

    range_im = np.max(im) - np.min(im);
    if range_im == 0:
        h = 0.1
    else:
        h = range_im * 0.001
    fp1 = im + h;
    g = minimum(fp1, fm)
    return 1 - reconstruction(1 - fm, 1 - g)

def complement(a):
    return a.max()-a

def regional_minima(a, connectivity=3):
    """Find the regional minima in an ndarray."""
    values = unique(a)
    delta = (values - minimum_filter(values, footprint=ones(connectivity)))[1:].min()
    marker = complement(a)
    mask = marker+delta
    return marker ==reconstruction(marker, mask,method='dilation',selem=np.ones((3, 3)))


def water_shed(img):
    dist = -ndi.distance_transform_edt(1 - np.logical_not(img))
    mask1 = imextendedmin(dist, 2)
    distance = -imimposemin(dist, mask1)
    local_maxi = regional_minima(-distance) * img
    local_maxi = remove_small_objects(local_maxi > 0, min_size=5, connectivity=1)
    markers, num_labels = ndi.label(local_maxi)
    labels = watershed(-distance, markers, mask=img)
    return labels, np.max(labels)

def find_max_area(labels,num_labels):
    max_area=0
    for i in range(1,num_labels+1):
        object_area=np.sum((labels==i).astype('float64'))
        if max_area<object_area:
            max_area=object_area
    return max_area

def remove_objects(labels, num_labels, min_size=20):
    for i in range(1, num_labels + 1):
        object_area = np.sum((labels == i).astype('float64'))
        if object_area < min_size:
            labels[labels == i] = 0

    return labels > 0

def get_colored_segmentation_image(seg_arr, n_classes, colors=class_colors):
    output_height = seg_arr.shape[0]
    output_width = seg_arr.shape[1]

    seg_img = np.zeros((output_height, output_width, 3))

    for c in range(n_classes):
        seg_arr_c = seg_arr[:, :] == c
        seg_img[:, :, 0] += ((seg_arr_c)*(colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((seg_arr_c)*(colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((seg_arr_c)*(colors[c][2])).astype('uint8')

    return seg_img

def overlay_seg_image(inp_img, seg_img):
    orininal_h = inp_img.shape[0]
    orininal_w = inp_img.shape[1]
    if len(inp_img.shape)==2:
        inp_img=cv2.cvtColor(inp_img, cv2.COLOR_GRAY2BGR)
    seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))

    fused_img = (inp_img/2 + seg_img/2).astype('uint8')
    return fused_img

def get_legends(class_names, colors=class_colors):

    n_classes = len(class_names)
    legend = np.zeros(((len(class_names) * 25) + 25, 125, 3),
                      dtype="uint8") + 255

    class_names_colors = enumerate(zip(class_names[:n_classes],
                                       colors[:n_classes]))

    for (i, (class_name, color)) in class_names_colors:
        color = [int(c) for c in color]
        cv2.putText(legend, class_name, (5, (i * 25) + 17),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
        cv2.rectangle(legend, (100, (i * 25)), (125, (i * 25) + 25),
                      tuple(color), -1)

    return legend

def concat_lenends(seg_img, legend_img):

    new_h = np.maximum(seg_img.shape[0], legend_img.shape[0])
    new_w = seg_img.shape[1] + legend_img.shape[1]

    out_img = np.zeros((new_h, new_w, 3)).astype('uint8') + legend_img[0, 0, 0]

    out_img[:legend_img.shape[0], :  legend_img.shape[1]] = np.copy(legend_img)
    out_img[:seg_img.shape[0], legend_img.shape[1]:] = np.copy(seg_img)

    return out_img

def visualize_segmentation(seg_arr, inp_img=None, n_classes=None,
                           colors=class_colors, class_names=None,
                           overlay_img=False, show_legends=False,
                           prediction_width=None, prediction_height=None):

    if n_classes is None:
        n_classes = np.max(seg_arr)+1

    seg_img = get_colored_segmentation_image(seg_arr, n_classes, colors=colors)

    if inp_img is not None:
        orininal_h = inp_img.shape[0]
        orininal_w = inp_img.shape[1]
        seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))

    if (prediction_height is not None) and (prediction_width is not None):
        seg_img = cv2.resize(seg_img, (prediction_width, prediction_height))
        if inp_img is not None:
            inp_img = cv2.resize(inp_img,
                                 (prediction_width, prediction_height))

    if overlay_img:
        assert inp_img is not None
        seg_img = overlay_seg_image(inp_img, seg_img)

    if show_legends:
        assert class_names is not None
        legend_img = get_legends(class_names, colors=colors)
        seg_img = concat_lenends(seg_img, legend_img)

    return seg_img

def preprocess(pil_img_prev, pil_curr, pil_mask, transforms):
    tensor_img_prev, tensor_img_curr, tensor_mask = transforms(pil_img_prev, pil_curr, pil_mask)
    return tensor_img_prev, tensor_img_curr, tensor_mask

def find_matchings(
        tracking_model, img_curr, img_prev, masks_curr, masks_prev, mask_curr, 
        centroids_curr, centroids_prev, cell_labels, last_new_label_index,
        device, transforms, min_size=20
    ):
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
        mask_pred = tracking_model(torch.unsqueeze(tensor_img_prev, 0), torch.unsqueeze(tensor_img_curr, 0))
        mask_pred = mask_pred.argmax(dim=1)
        mask_pred=mask_pred.cpu().numpy()
        mask_pred=remove_small_objects(mask_pred[0, :, :] > 0, min_size=min_size, connectivity=1).astype('float64')
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

def track_cells(
        pred_list, img_list, tracking_model, device, transforms, min_size=20
    ):
    tracking_model.eval()
    cell_labels,cell_centroids,tracked_imgs=[],[],[]
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
                matchings_result = find_matchings(
                    tracking_model, img_curr, img_prev, masks_curr, masks_prev, 
                    mask_curr, centroids_curr, centroids_prev, cell_labels, 
                    last_new_label_index, device, transforms, min_size=min_size
                )
                labels_curr, masks_curr, centroids_curr, last_new_label_index = (
                    matchings_result
                )
            
            tracked_imgs.append(add_labels_to_image(img_curr, labels_curr, centroids_curr))
            cell_labels.append(labels_curr)
            cell_centroids.append((np.array(centroids_curr)).tolist())
            centroids_prev = centroids_curr
            img_prev = img_curr
            masks_prev = masks_curr
            pbar.update(1)

    return cell_labels,cell_centroids,tracked_imgs