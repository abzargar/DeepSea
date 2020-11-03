from types import MethodType
from keras.models import *
from keras.layers import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import six
import numpy as np
from PIL import Image
import os
import cv2
from keras.callbacks import ModelCheckpoint,Callback,EarlyStopping
from .deepsea_tracker_data_loader import image_segmentation_generator, \
    verify_segmentation_dataset,class_colors,get_image_array
        
IMAGE_ORDERING="channels_last"


def add_batch_norm(conv_in,axis=3,activation='relu'):
    conv_out = BatchNormalization(axis=3)(conv_in)
    conv_out = Activation(activation)(conv_out)
    return conv_out

def res_block(inp,depth,dropout=0.2):
    conv = Conv2D(depth, (3, 3), padding='same')(inp)
    conv =add_batch_norm(conv)
    conv = Dropout(dropout)(conv)
    conv = Conv2D(depth, (3, 3) , padding='same')(conv)
    sc=Conv2D(depth, (1, 1), padding='same')(inp)
    conv = Add()([conv, sc])
    return conv

def get_segmentation_model(img_input, output):

    o_shape = Model(img_input, output).output_shape
    i_shape = Model(img_input, output).input_shape[0]
            
    if IMAGE_ORDERING == 'channels_first':
        output_height = o_shape[2]
        output_width = o_shape[3]
        input_height = i_shape[2]
        input_width = i_shape[3]
        n_classes = o_shape[1]
        output = (Permute((2, 1)))(output)
    elif IMAGE_ORDERING == 'channels_last':
        output_height = o_shape[1]
        output_width = o_shape[2]
        input_height = i_shape[1]
        input_width = i_shape[2]
        n_classes = o_shape[3]
    output = (Activation('softmax'))(output)
    model = Model(inputs=img_input, outputs=[output])
    model.output_width = output_width
    model.output_height = output_height
    model.n_classes = n_classes
    model.input_height = input_height
    model.input_width = input_width
    model.model_name = ""
    model.train = MethodType(train, model)
    model.predict_segmentation = MethodType(predict, model)
    model.predict_multiple = MethodType(predict_multiple, model)

    return model

def train(model,
          train_images,
          train_annotations,
          input_height=None,
          input_width=None,
          n_classes=None,
          verify_dataset=True,
          epochs=5,
          batch_size=2,
          validate=False,
          val_images=None,
          val_annotations=None,
          val_batch_size=2,
          load_weights=None,
          steps_per_epoch=512,
          val_steps_per_epoch=512,
          optimizer_name='adadelta',
          early_stopping=False,
          early_stopping_monitor='val_loss',
          early_stopping_patience=5,
          early_stopping_min_delta=.0005,
          metrics=['accuracy'],
          loss = 'categorical_crossentropy',
          plot_history=False,
          check_point_save=True,
          check_point_save_path='',
          model_name_to_save='tracker_last',
          ):

    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    output_height = model.output_height
    output_width = model.output_width

    if validate:
        assert val_images is not None
        assert val_annotations is not None

    if optimizer_name is not None:
        model.compile(loss=loss,
                      optimizer=optimizer_name,
                      metrics=metrics)
        
    if load_weights is not None and len(load_weights) > 0:
        print("Loading weights from ", load_weights)
        model.load_weights(load_weights)

    if verify_dataset:
        print("Verifying training dataset")
        verified = verify_segmentation_dataset(train_images,
                                               train_annotations,
                                               n_classes)
        assert verified
        if validate:
            print("Verifying validation dataset")
            verified = verify_segmentation_dataset(val_images,
                                                   val_annotations,
                                                   n_classes)
            assert verified

    train_gen = image_segmentation_generator(
        train_images, train_annotations, batch_size,  n_classes,
        input_height, input_width, output_height, output_width)

    if validate:
        val_gen = image_segmentation_generator(
            val_images, val_annotations,  val_batch_size,
            n_classes, input_height, input_width, output_height, output_width)
    
    
    callbacks = []
    
    if check_point_save:
        callbacks.append(ModelCheckpoint(os.path.join(check_point_save_path, model_name_to_save+'.hdf5'), monitor='val_loss',verbose=1, save_best_only=True))# set callback to save the best model correspondint to the minimum val loss
        
    
    if early_stopping:
        callbacks.append(EarlyStopping(monitor=early_stopping_monitor, min_delta=early_stopping_min_delta, patience=early_stopping_patience, verbose=1))

    if not validate:
        history=model.fit_generator(train_gen, steps_per_epoch,
                            epochs=epochs, callbacks=callbacks)
    else:
        history=model.fit_generator(train_gen,
                            steps_per_epoch,
                            validation_data=val_gen,
                            validation_steps=val_steps_per_epoch,
                            epochs=epochs, callbacks=callbacks)
    
    if plot_history:
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'validation'], loc='upper left')
        # plt.savefig(os.path.join(save_path, 'loss_graph.png'))
        
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


def overlay_seg_image(inp_img, seg_img):
    orininal_h = inp_img.shape[0]
    orininal_w = inp_img.shape[1]
    seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))

    fused_img = (inp_img/2 + seg_img/2).astype('uint8')
    return fused_img


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
        n_classes = np.max(seg_arr)

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


def predict(model=None, inp=None, out_fname=None,
            checkpoints_path=None, overlay_img=False,
            class_names=None, show_legends=False, colors=class_colors,
            prediction_width=None, prediction_height=None):

    assert (inp is not None)
    assert (((type(inp[0]) is np.ndarray) and (type(inp[1]) is np.ndarray)) 
            or (isinstance(inp[0], six.string_types) and isinstance(inp[1], six.string_types))),\
        "Input should be the image or the input file name"

    if isinstance(inp[0], six.string_types):
        inp_curr = Image.open(inp[0])
        inp_curr=np.array(inp_curr)
    if isinstance(inp[1], six.string_types):
        inp_prev = Image.open(inp[1])
        inp_prev=np.array(inp_prev)
           
    output_width = model.output_width
    output_height = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes

    x = get_image_array([inp_curr,inp_prev], input_width, input_height,reshape=True)
    pr = model.predict([np.array([x[0]]),np.array([x[1]])])[0]
    pr = pr.reshape((output_height,  output_width, n_classes)).argmax(axis=2)
    

    if out_fname is not None:
        seg_img = visualize_segmentation(pr, inp_prev, n_classes=n_classes,
                                      colors=colors, overlay_img=overlay_img,
                                      show_legends=show_legends,
                                      class_names=class_names,
                                      prediction_width=prediction_width,
                                      prediction_height=prediction_height)
        cv2.imwrite(out_fname, seg_img)
        
    return pr


def predict_multiple(model=None, inps=None, inp_dir=None, out_dir=None,
                     checkpoints_path=None, overlay_img=False,
                     class_names=None, show_legends=False, colors=class_colors,
                     prediction_width=None, prediction_height=None):


    if inps is None and (inp_dir[0] is not None):
        inps_curr = glob.glob(os.path.join(inp_dir[0], "*.jpg")) + glob.glob(
            os.path.join(inp_dir[0], "*.png")) + \
            glob.glob(os.path.join(inp_dir[0], "*.jpeg"))
        inps_curr = sorted(inps_curr)
        
    if inps is None and (inp_dir[1] is not None):
        inps_prev = glob.glob(os.path.join(inp_dir[1], "*.jpg")) + glob.glob(
            os.path.join(inp_dir[1], "*.png")) + \
            glob.glob(os.path.join(inp_dir[1], "*.jpeg"))
        inps_prev = sorted(inps_prev)    

    assert type(inps_curr) is list
    assert type(inps_prev) is list
    
    all_prs = []

    for i, (inp_curr,inp_prev) in enumerate(tqdm(zip(inps_curr,inps_prev))):
        if out_dir is None:
            out_fname = None
        else:
            if isinstance(inp_curr, six.string_types):
                out_fname = os.path.join(out_dir, os.path.basename(inp_curr))
            else:
                out_fname = os.path.join(out_dir, str(i) + ".jpg")

        pr = predict(model, [inp_curr,inp_prev], out_fname,
                     overlay_img=overlay_img, class_names=class_names,
                     show_legends=show_legends, colors=colors,
                     prediction_width=prediction_width,
                     prediction_height=prediction_height)

        all_prs.append(pr)

    return all_prs        

def measure_detection_metrics(pr_labels_list,gt_labels_list):
    
    tp,fp,fn=0,0,0
    for pr_lable_list,gt_label_list in zip(pr_labels_list,gt_labels_list):
        pr_labels,pr_num_labels = pr_lable_list
        gt_labels, gt_num_labels = gt_label_list
        IOU=[]
        flag=0
        for gt_label_id in range(1,gt_num_labels+1):
            mask1=(gt_labels==gt_label_id).astype('float64')
            mask2=mask1*(pr_labels>0).astype('float64')
            IOU.append(np.sum(mask1*mask2)/(np.sum(mask1+mask2)-np.sum(mask1*mask2)))
              
        if np.mean(IOU)>0.2:
            flag=1
            tp+=1
        else:
            flag=1
            fn+=1
        for pr_label_id in range(1,pr_num_labels+1): 
            if np.sum((pr_labels==pr_label_id).astype('float64')*(gt_labels>0).astype('float64'))==0:
               fp+=1 
            
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    Fscore=2*(precision*recall)/(precision+recall)
    
    results={
        "precision": precision,
        "recall": recall,
        "Fscore": Fscore
    }
    return results