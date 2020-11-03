from utils.deepsea_tracker_model import deepsea_tracker
import utils
from keras.optimizers import Adam
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model=deepsea_tracker()

cce_dice_loss = utils.losses.CategoricalCELoss() + utils.losses.DiceLoss()
metrics=[utils.metrics.IOUScore(class_weights=[0,1])]#Should be later multiplied by 2, only focus on foreground

model_name_to_save='tracker_last'
save_path=''
dataset_dir='dataset_for_cell_tracking'
    
model.train(
    train_images =  [dataset_dir+"/train_set/image_curr/",dataset_dir+"/train_set/image_prev/"],
    train_annotations = dataset_dir+"/train_set/label/",
    epochs=100,
    batch_size=8,
    validate=True,
    val_images=[dataset_dir+"/validation_set/image_curr/",dataset_dir+"/validation_set/image_prev/"],
    val_annotations=dataset_dir+"/validation_set/label/",
    val_batch_size=8,
    steps_per_epoch=2501,
    val_steps_per_epoch=417,
    optimizer_name=Adam(1e-4),
    early_stopping=True,
    metrics=metrics,
    loss=cce_dice_loss,
    plot_history=True,
    model_name_to_save=model_name_to_save
)

#load trained model (best model saved by early stopping)
model.load_weights(os.path.join(save_path, model_name_to_save+'.hdf5'))

# Predict model output for single image input
out = model.predict_segmentation(
    inp=[dataset_dir+"/test_set/image_curr/img_9.png",dataset_dir+"/test_set/image_prev/img_9.png"],
    out_fname="./tmp/out.png"
)

# Predict model output for multiple image inputs
out = model.predict_multiple(
    inp_dir=[dataset_dir+"/test_set/image_curr/",dataset_dir+"/test_set/image_prev/"],
    out_dir="./tmp/predict/"
)


