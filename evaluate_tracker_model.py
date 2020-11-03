from utils.deepsea_tracker_model import deepsea_tracker
from scipy import ndimage as ndi
import os
import cv2
from scipy.spatial import distance
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from utils.deepsea_tracker_data_loader import get_image_array, get_segmentation_array, get_pairs_from_paths
from utils.deepsea_tracker_utils import predict,measure_detection_metrics    
from skimage.morphology import remove_small_objects
from keras.backend.tensorflow_backend import set_session
import skimage.measure as measure

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)


model=deepsea_tracker(n_classes=2)


load_path='trained_models/'
model_name_to_load='tracker_last'
model.load_weights(os.path.join(load_path, model_name_to_load+'.hdf5'))


inp_prev_images_dir="dataset_for_cell_tracking/test_set/image_prev/"
inp_curr_images_dir="dataset_for_cell_tracking/test_set/image_curr/"
cell_annotations_dir="dataset_for_cell_tracking/test_set/label/"

cell_paths = get_pairs_from_paths(images_path=[inp_curr_images_dir,inp_prev_images_dir], segs_path=cell_annotations_dir)
paths = list(zip(*cell_paths))
inp_images = list(paths[0])
annotations = list(paths[1])

assert type(inp_images) is list
assert type(annotations) is list

pr_labels_list,pr_labels_list_easy,pr_labels_list_birth,pr_labels_list_hard,pr_labels_list_single=[],[],[],[],[]
gt_labels_list,gt_labels_list_easy,gt_labels_list_birth,gt_labels_list_hard,gt_labels_list_single=[],[],[],[],[]
pr_list=[]
gt_list=[]
for [inp_curr,inp_prev], ann in tqdm(zip(inp_images, annotations)):
        pr = predict(model,[inp_curr,inp_prev])
        pr=remove_small_objects(pr>0, min_size=50, connectivity=1).astype('int')
        
        _,img_curr_num_labels=ndi.label(cv2.imread(inp_curr)>0)
        pr_labels, pr_num_labels = ndi.label(pr)
        gt = get_segmentation_array(ann, model.n_classes,
                                    model.output_width, model.output_height)
        gt = gt.argmax(-1)
        gt=remove_small_objects(gt>0, min_size=10, connectivity=1).astype('int')
        gt_labels, gt_num_labels = ndi.label(gt)

        pr_labels_list.append((pr_labels,pr_num_labels))
        gt_labels_list.append((gt_labels, gt_num_labels))

        if gt_num_labels==1:
            pr_labels_list_single.append((pr_labels,pr_num_labels))
            gt_labels_list_single.append((gt_labels, gt_num_labels))
        
        elif gt_num_labels==2:
            pr_labels_list_birth.append((pr_labels,pr_num_labels))
            gt_labels_list_birth.append((gt_labels, gt_num_labels))
        if img_curr_num_labels==1:
            pr_labels_list_easy.append((pr_labels,pr_num_labels))
            gt_labels_list_easy.append((gt_labels, gt_num_labels))
        else:
            pr_labels_list_hard.append((pr_labels,pr_num_labels))
            gt_labels_list_hard.append((gt_labels, gt_num_labels))

print("\nWait...")
results=measure_detection_metrics(pr_labels_list,gt_labels_list)
print("\nOveral")
print(results)

print("\nWait...")
results=measure_detection_metrics(pr_labels_list_easy,gt_labels_list_easy)
print("\nEasy")
print(results)


print("\nWait...")
results=measure_detection_metrics(pr_labels_list_hard,gt_labels_list_hard)
print("\nHard")
print(results)

print("\nWait...")
results=measure_detection_metrics(pr_labels_list_single,gt_labels_list_single)
print("\nSingle tracking")
print(results)


print("\nWait...")
results=measure_detection_metrics(pr_labels_list_birth,gt_labels_list_birth)
print("\nBirth tracking")
print(results)









