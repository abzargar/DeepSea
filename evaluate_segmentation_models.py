from utils.deepsea_detector_model import deepsea_detector
from scipy import ndimage as ndi
import os
import cv2
import utils
from scipy.spatial import distance
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from utils.deepsea_detector_data_loader import  get_segmentation_array, get_pairs_from_paths 
from utils.deepsea_detector_utils import water_shed,find_max_area,remove_objects,predict, \
    measure_detection_metrics,measure_segmentation_metrics    
from keras.backend.tensorflow_backend import set_session
import skimage.measure as measure
from skimage.morphology import remove_small_objects

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)


model=deepsea_detector(n_classes=2) 

load_path='trained_models/'
model_name_to_load='cell_detection_model'
model.load_weights(os.path.join(load_path, model_name_to_load+'.hdf5'))


inp_images_dir="dataset_for_cell_segmentation/test_set/image/"
cell_annotations_dir="dataset_for_cell_segmentation/test_set/label/"

pre_process=False #False or True

cell_paths = get_pairs_from_paths(images_path=inp_images_dir, segs_path=cell_annotations_dir)
paths = list(zip(*cell_paths))
inp_images = list(paths[0])
annotations = list(paths[1])

assert type(inp_images) is list
assert type(annotations) is list


pr_list,pr_list_easy,pr_list_moderate,pr_list_hard=[],[],[],[]
gt_list,gt_list_easy,gt_list_moderate,gt_list_hard=[],[],[],[]
pr_labels_list,pr_labels_list_easy,pr_labels_list_moderate,pr_labels_list_hard=[],[],[],[]
gt_labels_list,gt_labels_list_easy,gt_labels_list_moderate,gt_labels_list_hard=[],[],[],[]
for inp, ann in tqdm(zip(inp_images, annotations)):
        pr = predict(model, inp,pre_process=pre_process)
        
        if np.sum(pr)>40:
            labels,num_labels = water_shed(pr)
            max_area=find_max_area(labels,num_labels)   
            pr=remove_objects(labels,num_labels, min_size=0.02*max_area)
            pr_labels,pr_num_labels = water_shed(pr)
        else:
            pr_labels, pr_num_labels = ndi.label(pr)

        gt = get_segmentation_array(ann, model.n_classes,
                                    model.output_width, model.output_height)
        gt = gt.argmax(-1)
        gt=remove_small_objects(gt>0, min_size=10, connectivity=1).astype('int')
        gt_labels, gt_num_labels = ndi.label(gt)
        pr_list.append(pr)
        gt_list.append(gt)
        pr_labels_list.append((pr_labels,pr_num_labels))
        gt_labels_list.append((gt_labels, gt_num_labels))
        if gt_num_labels<10:
            pr_list_easy.append(pr)
            gt_list_easy.append(gt)
            pr_labels_list_easy.append((pr_labels,pr_num_labels))
            gt_labels_list_easy.append((gt_labels, gt_num_labels))
        
        elif(gt_num_labels>=10 and gt_num_labels<20):
            pr_list_moderate.append(pr)
            gt_list_moderate.append(gt)
            pr_labels_list_moderate.append((pr_labels,pr_num_labels))
            gt_labels_list_moderate.append((gt_labels, gt_num_labels))
        else:
            pr_list_hard.append(pr)
            gt_list_hard.append(gt)
            pr_labels_list_hard.append((pr_labels,pr_num_labels))
            gt_labels_list_hard.append((gt_labels, gt_num_labels))


results=measure_segmentation_metrics(pr_list,gt_list)   
print("\nOveral")
print(results)

results=measure_segmentation_metrics(pr_list_easy,gt_list_easy)   
print("\nEasy")
print(results)

results=measure_segmentation_metrics(pr_list_moderate,gt_list_moderate)   
print("\nModerate")
print(results)


results=measure_segmentation_metrics(pr_list_hard,gt_list_hard)   
print("\nHard")
print(results)

print("\nWait...")
results=measure_detection_metrics(pr_labels_list,gt_labels_list)
print("\nOveral")
print(results)

print("\nWait...")
results=measure_detection_metrics(pr_labels_list_easy,gt_labels_list_easy)
print("\nEasy")
print(results)

print("\nWait...")
results=measure_detection_metrics(pr_labels_list_moderate,gt_labels_list_moderate)
print("\nModerate")
print(results)

print("\nWait...")
results=measure_detection_metrics(pr_labels_list_hard,gt_labels_list_hard)
print("\nHard")
print(results)







