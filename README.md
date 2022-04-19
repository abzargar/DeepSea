# DeepSea 

This repo contains the training and evaluation code for the project [DeepSea: An efficient deep learning model for automated cell segmentation and tracking](https://www.deepseas.org/). 

This work presents a versatile and trainable deep-learning-based software, termed DeepSea, that allows for both segmentation and tracking of single cells and their nuclei in sequences of phasecontrast live microscopy images.


### Datasets

To download our datasets go to https://deepseas.org/datasets/ or:

* Link to [Original annotated dataset](https://drive.google.com/drive/folders/13RhhBAetSWkjySyhJcDqj_FaO09hxkhO?usp=sharing)

* Link to [dataset for cell segmentation](https://drive.google.com/drive/folders/1gJIkwUQEtut4JCCoUXUcKUWp2gVYxQ9P?usp=sharing)

* Link to [dataset for cell tracking](https://drive.google.com/drive/folders/17n0Ex8NQS-REB5ZAMlntVnYBnSmZJtLR?usp=sharing)

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
python train_segmentation.py --train_dir  segmentation_dataset/train/  --lr 0.001 --max_epoch 200 --batch_size 32 --output_dir tmp/
```
* #### Run the cell tracker training
Run train_tracker.py with the train set of [tracking dataset](https://drive.google.com/drive/folders/1iCC22iz7UBQdmADLuDe8ugAkmUqqsv13?usp=sharing)
```
Example:
python train_tracker.py --train_dir tracking_dataset/train/ --lr 0.001 --max_epoch 200 --batch_size 32 --output_dir tmp/
```
* #### Run the cell segmentation test
Run test_segmentation.py with the test set of segmentation dataset and trained segmentation model
```
Example:
python test_segmentation.py --test_dir segmentation_dataset/test/ --ckpt_dir trained_models/segmentation.pth --output_dir tmp/
```
* #### Run the cell segmentation test
Run test_tracker.py with the test set of tracking dataset and trained tracker model
```
Example:
python test_tracker.py --test_dir tracking_dataset/test --ckpt_dir trained_models/tracker.pth --output_dir tmp/
```
* #### Measure MOTA
Run measure_MOTA.py with a time-lapse microscopy set and both segmentation and tracker models
```
Example:
python measure_MOTA.py --test_dir tracking_dataset/test/set_9_MC2C12/ --seg_ckpt_dir trained_models/segmentation.pth --tracker_ckpt_dir trained_models/tracker.pth  --output_dir tmp/
```
### Useful Information
If you have any questions, contact us at abzargar@ucsc.edu.

### Acknowledgements
This work was supported by the NIGMS/NIH through a Pathway to Independence Award 435 K99GM126027 (S.A.S.) and start- up package of the University of California, Santa Cruz