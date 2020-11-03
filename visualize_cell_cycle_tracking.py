from utils.deepsea_detector_model import deepsea_detector
from utils.deepsea_tracker_model import deepsea_tracker
from utils.deepsea_detector_utils import detect_and_track_cells
import numpy as np
import os
   
model_cell=deepsea_detector(n_classes=2) 
model_tracker=deepsea_tracker(n_classes=2)
 
load_path='trained_models/'
model_name_to_load='cell_detection_model'
model_cell.load_weights(os.path.join(load_path, model_name_to_load+'.hdf5'))

load_path='trained_models/'
model_name_to_load='tracker_model'
model_tracker.load_weights(os.path.join(load_path, model_name_to_load+'.hdf5'))

detect_and_track_cells(model_cell,model_tracker, inp_images_dir="dataset_for_cell_cycle_tracking/test_set/A11/original_images/" ,save_path="tmp/tracked_images/")
