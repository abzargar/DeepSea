from collections import defaultdict
from utils.deepsea_tracker_model import deepsea_tracker
from utils.deepsea_detector_model import deepsea_detector
from scipy import ndimage as ndi
from munkres import Munkres
import os
from scipy.spatial import distance
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from utils.deepsea_detector_data_loader import get_image_array
from utils.deepsea_detector_utils import detect_cells,find_tracks 
from keras.backend.tensorflow_backend import set_session
import skimage.measure as measure


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)


dataset_dir="dataset_for_cell_cycle_tracking/test/"


model_tracker=deepsea_tracker(n_classes=2)
model_cell=deepsea_detector(n_classes=2)

load_path='trained_models/'
model_name_to_load='cell_detection_model'
model_cell.load_weights(os.path.join(load_path, model_name_to_load+'.hdf5'))

load_path='trained_models/'
model_name_to_load='tracker_last3'
model_tracker.load_weights(os.path.join(load_path, model_name_to_load+'.hdf5'))

pad_size=15#to ignore objects close to the ede of image
original_images_width=1344
original_images_height=1024

def get_images_from_path(images_path):

        ACCEPTABLE_IMAGE_FORMATS = [".tif",".jpg", ".jpeg", ".png", ".bmp"]

        image_files = []
        
        for dir_entry in os.listdir(images_path):
            if os.path.isfile(os.path.join(images_path, dir_entry)) and \
                    os.path.splitext(dir_entry)[1] in ACCEPTABLE_IMAGE_FORMATS:
                file_name, file_extension = os.path.splitext(dir_entry)
                image_files.append(os.path.join(images_path, dir_entry))
                
        return image_files

def get_labels_from_path(labels_path):

        ACCEPTABLE_IMAGE_FORMATS = [".txt"]
        
        label_files = []
        
        for dir_entry in os.listdir(labels_path):
            if os.path.isfile(os.path.join(labels_path, dir_entry)) and \
                    os.path.splitext(dir_entry)[1] in ACCEPTABLE_IMAGE_FORMATS:
                file_name, file_extension = os.path.splitext(dir_entry)
                label_files.append(os.path.join(labels_path, dir_entry))
                
        return label_files    
        

def  detect_and_track(image_files,model_cell,model_tracker):
        
        cell_labels=[]
        cell_centroids=[]
        cell_labeled_imgs=[]
        tracked_images=[]
         
        for idx,inp in enumerate(image_files):
            print('frame_id->'+str(idx))
            centroids_curr,masks_curr,mask_curr,labels_color=detect_cells(model_cell,inp)
            
            img_curr=get_image_array(inp)
    
            
            if idx==0:
                labels_curr=[str(i) for i in range(len(centroids_curr))]
                last_new_label_index=len(centroids_curr)-1
            else:
                labels_curr,masks_curr,centroids_curr,last_new_label_index=find_tracks(model_tracker,img_curr,img_prev,masks_curr,masks_prev,mask_curr,mask_prev,centroids_curr,centroids_prev,labels_prev,last_new_label_index)
    
            cell_labels.append(labels_curr)
            cell_centroids.append((np.array(centroids_curr)-pad_size).tolist())
            centroids_prev=centroids_curr
            labels_prev=labels_curr
            mask_prev=mask_curr
            img_prev=img_curr
            masks_prev=masks_curr
        
        return cell_labels,cell_centroids

def read_labeles(label_files):
    cell_labels=[]
    cell_centroids=[]

    for idx,label_file_path in enumerate(label_files):
        file = open(label_file_path,"r")
        lines=file.readlines()
        lines.pop(0)
        labels,centroids=[],[]
        for line in lines:
            labels.append(line.replace('\n', '').split('\t')[0])
            centroids.append([int(i) for i in line.replace('\n', '').split('\t')[1:]])
        cell_labels.append(labels)
        cell_centroids.append(centroids)
    return  cell_labels,cell_centroids   
        
def remove_objects_from_list(cell_label,cell_centroid):
    centroid_remove_list,label_remove_list=[],[]
    for label,centroid in zip(cell_label,cell_centroid):
      if centroid[0]<pad_size or centroid[0]>(512-pad_size) or centroid[1]<pad_size or centroid[1]>(512-pad_size):
          centroid_remove_list.append(centroid)
          label_remove_list.append(label)
    for label,centroid in zip(label_remove_list,centroid_remove_list):
        cell_centroid.remove(centroid)
        cell_label.remove(label)
    return cell_label,cell_centroid  
      
def evaluate_tracking_results(cell_labels,cell_centroids,gt_cell_labels,gt_cell_centroids):
    seq_trajectories= defaultdict(list)
    gt_tracker      = []
    gt_id_switch     = []
    gt_fragmentation = []
    n_gt,n_tr=0,0
    max_cost = 1e9
    hm = Munkres()
    tp,fn,fp=0,0,0
    
    for idx,(cell_label,cell_centroid,gt_cell_label,gt_cell_centroid) in enumerate(zip(cell_labels,cell_centroids,gt_cell_labels,gt_cell_centroids)):

        for idx in range(len(gt_cell_centroid)):
            gt_cell_centroid[idx][1],gt_cell_centroid[idx][0]=np.round(gt_cell_centroid[idx][0]/original_images_width*512),np.round(gt_cell_centroid[idx][1]/original_images_height*512)
        
        gt_cell_label,gt_cell_centroid=remove_objects_from_list(gt_cell_label,gt_cell_centroid)
        
        cell_label,cell_centroid=remove_objects_from_list(cell_label,cell_centroid)      
        
        n_tr+=len(cell_label)
        n_gt+=len(gt_cell_label)    
        
        cost_matrix = []
        frame_ids = [[],[]]
        # loop over ground truth objects in one frame
        for idx,(gt_label,gt_centroid) in enumerate(zip(gt_cell_label,gt_cell_centroid)):

            frame_ids[0].append(gt_label)
            frame_ids[1].append(-1)
            gt_tracker.append(-1)
            gt_id_switch.append(0)
            gt_fragmentation.append(0)
            cost_row= []
            # loop over tracked objects in one frame
            for label,centroid in zip(cell_label,cell_centroid):

                dst=distance.euclidean(centroid, gt_centroid)
            
                if dst>=5:
                    cost_row.append(max_cost)
                else:
                    cost_row.append(1) 
            cost_matrix.append(cost_row)
            # all ground truth trajectories are initially not associated
            # extend groundtruth trajectories lists (merge lists)
            seq_trajectories[gt_label].append(-1)
        association_matrix = hm.compute(cost_matrix)
         # tmp variables for sanity checks
        tmptp = 0

        # mapping for tracker ids and ground truth ids
        for row, col in association_matrix:
            # apply gating on boxoverlap
            c = cost_matrix[row][col]
            if c < max_cost:
                seq_trajectories[gt_cell_label[row]][-1] = cell_label[col]
         
                # true positives are only valid associations
                tp += 1
                tmptp+= 1
            
        fn+= len(gt_cell_label)-len(association_matrix) 
        fp += len(cell_label) - tmptp
        mostly_lost,mostly_tracked,id_switches,fragments=0,0,0,0
    for g in seq_trajectories.values():
        # all frames of this gt trajectory are not assigned to any detections
        if all([this==-1 for this in g]):
            mostly_lost+=1
            continue
        # compute tracked frames in trajectory
        last_id = g[0]
        # first detection (necessary to be in gt_trajectories) is always tracked
        tracked = 1 if g[0]!=-1 else 0
        
        for f in range(1,len(g)):
            
            if last_id != g[f] and last_id != -1 and g[f] != -1 and g[f-1] != -1:
                id_switches += 1
            if f < len(g)-1 and g[f-1] != g[f] and last_id != -1 and g[f] != -1 and g[f+1] != -1:
                fragments += 1
            if g[f] != -1:
                tracked += 1
                last_id = g[f]
        # handle last frame; tracked state is handled in for loop (g[f]!=-1)
        if len(g)>1 and g[f-1] != g[f] and last_id != -1  and g[f] != -1:
            fragments += 1

        # compute mostly_tracked/partialy_tracked/mostly_lost
        tracking_ratio = tracked /len(g)
        if tracking_ratio > 0.8:
            mostly_tracked += 1
        elif tracking_ratio < 0.2:
            mostly_lost += 1
        
    
    number_of_trajectories=len(seq_trajectories)
    mostly_tracked /= number_of_trajectories
    mostly_lost /= number_of_trajectories
    
    if (fp+tp)==0 or (tp+fn)==0:
        recall = 0.
        precision = 0.
    else:
        recall = tp/(tp+fn)
        precision = tp/(fp+tp)
    
    MOTA=1-(fp+fn+id_switches)/n_gt
    results=np.array([MOTA,mostly_tracked,mostly_lost,
                      precision,recall,number_of_trajectories,
                      fragments,id_switches,n_gt],np.float64)
        
    return results
            

def main():
    
    results=np.zeros(9)    
    for dir_entry in os.listdir(dataset_dir):
        print('dir_entry->'+dir_entry)
        images_path=os.path.join(dataset_dir,dir_entry,'original_images')
        gt_labels_path=os.path.join(dataset_dir,dir_entry,'labels')
        image_files=get_images_from_path(images_path)
        
        cell_labels,cell_centroids=detect_and_track(image_files,model_cell,model_tracker)
        
        gt_label_files=get_labels_from_path(gt_labels_path)
        gt_cell_labels,gt_cell_centroids=read_labeles(gt_label_files)
        
        results=results+evaluate_tracking_results(cell_labels,cell_centroids,gt_cell_labels,gt_cell_centroids)
    results[0:5]=results[0:5]/len(os.listdir(dataset_dir))   
    print({
        "MOTA": results[0],
        "MT":results[1],
        "ML":results[2],
        "Precision": results[3],
        "Recall": results[4],
        "Number of trajectories":results[5],
        "Frag":results[6],
        "IDS":results[7],
        "Number of cells":results[8]
    })
if __name__ == "__main__":
    main()