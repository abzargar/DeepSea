# DeepSea 

This repo contains the training and evaluation code for the project [DeepSea: An efficient deep learning model for automated cell segmentation and tracking](https://www.deepseas.org/). 

This work presents a versatile and trainable deep-learning-based software, termed DeepSea, that allows for both segmentation and tracking of single cells in sequences of phase-contrast live microscopy images.


### Datasets

To download our datasets go to https://deepseas.org/datasets/ or:

* Link to [Original annotated dataset](https://drive.google.com/drive/folders/13RhhBAetSWkjySyhJcDqj_FaO09hxkhO?usp=sharing)

* Link to [dataset example for cell segmentation](https://drive.google.com/drive/folders/18odgkzafW8stHkzME_s7Es-ue7odVAc5?usp=sharing)

* Link to [dataset example for cell tracking](https://drive.google.com/drive/folders/10LWey85fgHgFj_myIr1CYSOviD4SleE4?usp=sharing)

### Pre-trained models
They are saved in the folder "trained_models".

### Requirements

* [Optional] Create a conda or python virtual environment.

* Install required packages using the `requirements.txt` file.
```
pip install -r requirements.txt
```

### Usage
* #### Run the cell segmentation training
Run train_segmentation.py with the train set of [segmentation dataset](https://drive.google.com/drive/folders/1iCC22iz7UBQdmADLuDe8ugAkmUqqsv13?usp=sharing)
```
Example:
python train_segmentation.py --train_set_dir  segmentation_dataset/train/  --lr 0.001 --max_epoch 200 --batch_size 32 --output_dir tmp/
```
* #### Run the cell tracker training
Run train_tracker.py with the train set of [tracking dataset](https://drive.google.com/drive/folders/1iCC22iz7UBQdmADLuDe8ugAkmUqqsv13?usp=sharing)
```
Example:
python train_tracker.py --train_set_dir tracking_dataset/train/ --lr 0.001 --max_epoch 200 --batch_size 32 --output_dir tmp/
```
* #### Run the cell segmentation test
Run evaluate_test_set_segmentation.py with the test set of segmentation dataset and segmentation model
```
Example:
python evaluate_test_set_segmentation.py --test_set_dir segmentation_dataset/test/ --ckpt_dir trained_models/segmentation.pth --output_dir tmp/
```
* #### Run the cell tracking test
Run evaluate_test_set_tracking.py with the test set of tracking dataset and tracking model
```
Example:
python evaluate_test_set_tracking.py --test_set_dir tracking_dataset/test --ckpt_dir trained_models/tracker.pth --output_dir tmp/
```
* #### Run the single image segmentation test
Run test_single_image_segmentation.py with one single cell image and segmentation model
```
Example:
python test_single_image_segmentation.py --single_img_dir segmentation_dataset/test/images/A11_z003_c001.png --ckpt_dir trained_models/segmentation.pth --output_dir tmp/
```
* #### Run the single image set tracking test
Run test_single_set_tracking.py with one single time-lapse microscopy image sequences and tracking model
```
Example:
python test_single_set_tracking.py --single_image_set_dir tracking_dataset/test/set_13_MESC/images/ --seg_ckpt_dir trained_models/segmentation.pth --tr_ckpt_dir trained_models/tracker.pth --output_dir tmp/
```
* #### Measure MOTA
Run measure_MOTA.py with a time-lapse microscopy set and both segmentation and tracking models
```
Example:
python measure_MOTA.py --single_test_set_dir tracking_dataset/test/set_13_MESC/ --seg_ckpt_dir trained_models/segmentation.pth --tr_ckpt_dir trained_models/tracker.pth  --output_dir tmp/
```
### DeepSea GUI Software
Our DeepSea software is available on https://deepseas.org/software/ 
with examples and instructions. DeepSea software is a user-friendly and automated software designed
to enable researchers to 1) load and explore their phase-contrast cell images in a 
high-contrast display, 2) detect and localize cell bodies using the pre-trained DeepSea 
segmentation model, 3) track and label cell lineages across the frame sequences using the pre-trained 
DeepSea tracking model, 4) manually correct the DeepSea models' outputs using user-friendly editing 
options, 5) train a new model with a new cell type dataset if needed, 6) save the results and cell label 
and feature reports on the local system. It employs our latest trained DeepSea models in the segmentation and tracking processes.
It employs our last trained DeepSea models in the segmentation and tracking processes.
![Screenshot](DeepSea_Software.png)

### Useful Information
If you have any questions, contact us at abzargar@ucsc.edu.

### Acknowledgements
This work was supported by the NIGMS/NIH through a Pathway to Independence Award 435 K99GM126027 (S.A.S.) and start- up package of the University of California, Santa Cruz