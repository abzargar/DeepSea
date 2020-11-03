from utils.deepsea_detector_model import deepsea_detector
import utils
from keras.optimizers import Adam
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


model=deepsea_detector()


cce_dice_loss = utils.losses.CategoricalCELoss() + utils.losses.DiceLoss()
metrics=[utils.metrics.IOUScore(class_weights=[0,1])]#Should be later multiplied by 2, only focus on foreground

model_name_to_save='detector_last'
save_path=''
dataset_dir='dataset_for_nucleus_segmentation'


model.train(
    train_images =  dataset_dir+"/train_set/image/",
    train_annotations = dataset_dir+"/train_set/label/",
    epochs=100,
    batch_size=2,
    validate=True,
    val_images=dataset_dir+"/validation_set/image/",
    val_annotations=dataset_dir+"/validation_set/label/",
    val_batch_size=3,
    steps_per_epoch=1611,
    val_steps_per_epoch=358,
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
    inp=dataset_dir+"/test_set/image/A11_z007_c001.png",
    out_fname="./tmp/out.png"
)

# Predict model output for multiple image inputs
out = model.predict_multiple(
    inp_dir=dataset_dir+"/test_set/image/",
    out_dir="./tmp/predict/"
)

# evaluating the model by some metrics
#training set
results=model.evaluate_segmentation( inp_images_dir=dataset_dir+"/train_set/image/"  , annotations_dir=dataset_dir+"/train_set/label/",pre_process=False  )
print(results)
#test set
results=model.evaluate_segmentation( inp_images_dir=dataset_dir+"/test_set/image/"  , annotations_dir=dataset_dir+"/test_set/label/",pre_process=False )
print(results)
