from types import MethodType
from keras.models import *
from keras.layers import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import six
import glob
from PIL import Image
import collections
from scipy.spatial import distance
from scipy.ndimage import grey_dilation, generate_binary_structure, minimum_filter
from skimage.segmentation import watershed
import skimage.measure as measure        
from numpy import unique,ones,minimum
from keras.callbacks import ModelCheckpoint,Callback,EarlyStopping
import numpy as np
import cv2
from scipy import ndimage as ndi
from .deepsea_detector_data_loader import image_segmentation_generator, \
    verify_segmentation_dataset,get_image_array,get_segmentation_array,class_colors, get_pairs_from_paths

from skimage.morphology import reconstruction,remove_small_objects


IMAGE_ORDERING="channels_last"

def complement(a):
    return a.max()-a

def regional_minima(a, connectivity=3):
    """Find the regional minima in an ndarray."""
    values = unique(a)
    delta = (values - minimum_filter(values, footprint=ones(connectivity)))[1:].min()
    marker = complement(a)
    mask = marker+delta
    return marker ==reconstruction(marker, mask,method='dilation',selem=np.ones((3, 3)))

def remove_objects(labels,num_labels, min_size=20):
    for i in range(1,num_labels+1):
        object_area=np.sum((labels==i).astype('float64'))
        if object_area<min_size:
            labels[labels==i]=0
    
    return labels>0

def find_max_area(labels,num_labels):
    max_area=0
    for i in range(1,num_labels+1):
        object_area=np.sum((labels==i).astype('float64'))
        if max_area<object_area:
            max_area=object_area
    return max_area 

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
    i_shape = Model(img_input, output).input_shape
            
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
    model = Model(img_input, output)
    model.output_width = output_width
    model.output_height = output_height
    model.n_classes = n_classes
    model.input_height = input_height
    model.input_width = input_width
    model.model_name = ""
    model.train = MethodType(train, model)
    model.predict_segmentation = MethodType(predict, model)
    model.predict_multiple = MethodType(predict_multiple, model)
    model.evaluate_segmentation = MethodType(evaluate, model)

    return model






def morphological_reconstruction(marker, mask, connectivity=3):
    """
    Perform morphological reconstruction of the marker into the mask.
    """
    sel = generate_binary_structure(marker.ndim, connectivity)
    
    diff = True
    while diff:
        markernew = grey_dilation(marker, footprint=sel)
        plt.figure(1)
        plt.imshow(markernew,cmap=plt.cm.gray)
        plt.figure(2)
        plt.imshow(mask,cmap=plt.cm.gray)
        markernew = minimum(markernew, mask)

        diff = (markernew-marker).max() > 0
        marker = markernew
    return marker

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
    fm = im.copy();
    fm[BW] = -1e4;
    
    fm[~BW] = 1e4;

    range_im = np.max(im) - np.min(im);
    if range_im == 0:
        h = 0.1
    else:
        h = range_im * 0.001    
    fp1 = im + h;
    
    g=minimum(fp1,fm)
    return 1-reconstruction(1-fm, 1-g)

def water_shed(img):
        dist=-ndi.distance_transform_edt(1-np.logical_not(img))
        mask1=imextendedmin(dist, 2)
        distance=-imimposemin(dist, mask1)
        local_maxi=regional_minima(-distance)*img
        local_maxi=remove_small_objects(local_maxi>0, min_size=5, connectivity=1)   
        markers, num_labels = ndi.label(local_maxi)
        labels = watershed(-distance, markers, mask=img)
        return labels,np.max(labels)

def detect_cells(model_cell,input_img,pr_labels=None,pre_process=True,pad_size=15):
        if pr_labels is None:
            pr = predict(model_cell, input_img,pre_process=pre_process)
        
            if np.sum(pr)>40:
                labels,num_labels = water_shed(pr)
                max_area=find_max_area(labels,num_labels)   
                pr=remove_objects(labels,num_labels, min_size=0.02*max_area)
                pr_labels,pr_num_labels = water_shed(pr)
            else:
                pr_labels, pr_num_labels = ndi.label(pr)
            
        centroids,masks = [],[]
        
        mask_curr=np.pad(pr_labels, (pad_size, pad_size), 'constant')
        for label_id in range(1,np.max(pr_labels)):
            mask=pr_labels==label_id
            rp=measure.regionprops(mask.astype('int'))[0]
            cell_centroid=list(np.round(rp.centroid).astype('int'))  
            centroids.append([cell_centroid[0]+pad_size,cell_centroid[1]+pad_size])
            mask=np.pad(mask, (pad_size, pad_size), 'constant') 
            masks.append(mask)

        return centroids,masks,mask_curr,pr_labels
    
def detect_nucleus(model_nucleus,input_img,pre_process=True):
        pr = predict(model_nucleus, input_img,pre_process=pre_process)
        
        if np.sum(pr)>40:
            labels,num_labels = water_shed(pr)
            max_area=find_max_area(labels,num_labels)   
            pr=remove_objects(labels,num_labels, min_size=0.02*max_area)
            pr_labels,pr_num_labels = water_shed(pr)
        else:
            pr_labels, pr_num_labels = ndi.label(pr)

        return pr_labels
    
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

def find_tracks(model_tracker,img_curr,img_prev,masks_curr,masks_prev,mask_curr,mask_prev,centroids_curr,centroids_prev,labels_prev,last_new_label_index,pad_size=15):
        labels_curr=[]
        centroids_curr_new=[]
        masks_curr_new=[]
        scores=[]
        mask_curr=mask_curr.astype('float64')
        img_prev=np.pad(img_prev, (pad_size, pad_size), 'constant')
        img_curr=np.pad(img_curr, (pad_size, pad_size), 'constant')
        
        argmin_list=[]
        for idx,mask_prev in enumerate(masks_prev):
            rp=measure.regionprops(mask_prev.astype('int'))
            bbox_prev=list(rp[0].bbox)
            mask_prev=mask_prev.astype('float64')
            bbox_prev[0],bbox_prev[1]=bbox_prev[0]-pad_size,bbox_prev[1]-pad_size
            bbox_prev[2],bbox_prev[3]=bbox_prev[2]+pad_size,bbox_prev[3]+pad_size
            ROI_mask_prev=(mask_prev*img_prev)[bbox_prev[0]:bbox_prev[2],bbox_prev[1]:bbox_prev[3]]
            ROI_mask_curr=((mask_curr>0).astype('float64')*img_curr)[bbox_prev[0]:bbox_prev[2],bbox_prev[1]:bbox_prev[3]]
            ROI_mask_prev = cv2.resize(ROI_mask_prev, (model_tracker.input_width, model_tracker.input_height))
            ROI_mask_curr = cv2.resize(ROI_mask_curr, (model_tracker.input_width, model_tracker.input_height))
   
            ROI_mask_prev=ROI_mask_prev.reshape((model_tracker.input_width, model_tracker.input_height,1))
            ROI_mask_curr=ROI_mask_curr.reshape((model_tracker.input_width, model_tracker.input_height,1))
            
            pr = model_tracker.predict([np.array([ROI_mask_curr]),np.array([ROI_mask_prev])])[0]
            pr = pr.reshape((model_tracker.output_height,  model_tracker.output_width, model_tracker.n_classes)).argmax(axis=2)
            pr=remove_small_objects(pr>0, min_size=50, connectivity=1).astype('float64')

            mask_curr_cropped=mask_curr[bbox_prev[0]:bbox_prev[2],bbox_prev[1]:bbox_prev[3]]
            labels_idx=np.unique(np.round(mask_curr_cropped))
            labels_idx=labels_idx[labels_idx!=0]
            if np.sum(mask_curr_cropped):
                
                for j in labels_idx:
                        markers=np.round(cv2.resize((mask_curr_cropped==j).astype('float64'),(128,128)))
                        rp=measure.regionprops(markers.astype('int'))
                        IOU=np.sum(markers*pr)/np.sum(markers)
                        if IOU>0.2:
                            center=np.round(rp[0].centroid).astype('int')
                            center[0],center[1]=center[0]*(bbox_prev[2]-bbox_prev[0])/128+bbox_prev[0],center[1]*(bbox_prev[3]-bbox_prev[1])/128+bbox_prev[1]
                            dst_list=[distance.euclidean(center, centroid_curr) for centroid_curr in centroids_curr]
                            if np.min(dst_list)<5:
                                labels_curr.append(labels_prev[idx])
                                centroids_curr_new.append(centroids_curr[np.argmin(dst_list)])
                                scores.append(IOU)
                                masks_curr_new.append(masks_curr[np.argmin(dst_list)])
                                argmin_list.append(np.argmin(dst_list))
                
            
        for idx in range(len(centroids_curr)):
            if idx not in argmin_list:
                last_new_label_index=last_new_label_index+1
                labels_curr.append(str(last_new_label_index))
                centroids_curr_new.append(centroids_curr[idx])
                masks_curr_new.append(masks_curr[idx])
            
                      
        duplicates_centroids=collections.defaultdict(list)
        for idx,centroid in enumerate(centroids_curr_new):
            if centroids_curr_new.count(centroid) > 1:
              duplicates_centroids[(centroid[0],centroid[1])].append(idx)

        if duplicates_centroids: 
            remove_list=[]
            for centroid in duplicates_centroids:
                label_1=labels_curr[duplicates_centroids[centroid][0]]
                label_2=labels_curr[duplicates_centroids[centroid][1]]
                centroid_1=centroids_prev[labels_prev.index(label_1)]
                centroid_2=centroids_prev[labels_prev.index(label_2)]
                dst1=distance.euclidean(centroid, centroid_1)
                dst2=distance.euclidean(centroid, centroid_2)
                if dst1<dst2:
                   remove_list.append(duplicates_centroids[centroid][1])
                else:
                   remove_list.append(duplicates_centroids[centroid][0]) 
            
            if remove_list:
                centroids_curr_new,masks_curr_new,labels_curr=remove_detected_objects(remove_list,centroids_curr_new,masks_curr_new,labels_curr)
        
        
        duplicates_labels=collections.defaultdict(list)
        duplicates_labels_scores=collections.defaultdict(list)
        for idx,label in enumerate(labels_curr):
            if labels_curr.count(label) > 1:
              duplicates_labels[label].append(idx)
              duplicates_labels_scores[label].append(scores[idx])


        if duplicates_labels: 
            remove_list=[]
            for label_idx,label in enumerate(duplicates_labels):
                flag=0    
                for idx,center in enumerate(centroids_curr_new):
                    if idx in duplicates_labels[label]:
                        continue
                    for j in range(len(duplicates_labels[label])):
                        dst=distance.euclidean(centroids_curr_new[duplicates_labels[label][j]], center)
                    
                        if dst<5 :
                            flag=1
                            if scores[duplicates_labels[label][j]]<scores[idx]:

                                remove_list.append(duplicates_labels[label][j])
                        
                if flag==0:        
                    labels_curr[duplicates_labels[label][0]]=label+'_1'
                    labels_curr[duplicates_labels[label][1]]=label+'_2'
            
            if remove_list:
                 centroids_curr_new,masks_curr_new,labels_curr=remove_detected_objects(remove_list,centroids_curr_new,masks_curr_new,labels_curr)

        return labels_curr,masks_curr_new,centroids_curr_new,last_new_label_index
    
def add_labels_to_image(img,lebels,centroids,save_path=None,fontScale=0.4,thickness=1):
    
    if isinstance(img, str): 
        img=(get_image_array(img)*255).astype('uint8')
    if len(img.shape)==2:
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        
    for idx,label in enumerate(lebels):
       point=[x + y for x, y in zip(centroids[idx][::-1], [-5,3])]
       img=cv2.putText(img, label, tuple(point), cv2.FONT_HERSHEY_SIMPLEX, fontScale,(100,255,50), thickness);
    # if save_path is not None:
    #     cv2.imwrite(os.path.join(save_path, img.split('/')[-1]), img)
    return img    
    

def train(model,
          train_images,
          train_annotations,
          input_height=None,
          input_width=None,
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
          model_name_to_save='detector_last',
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
        train_images, train_annotations,  batch_size,  n_classes,
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
            prediction_width=None, prediction_height=None,pre_process=True):

    assert (inp is not None)
    assert ((type(inp) is np.ndarray) or isinstance(inp, six.string_types)),\
        "Input should be the image or the input file name"

    if isinstance(inp, six.string_types):
        inp = Image.open(inp)
        inp=np.array(inp)
        
    output_width = model.output_width
    output_height = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes

    x = get_image_array(inp, input_width, input_height,
                        reshape=True,pre_process=pre_process)
    pr = model.predict(np.array([x]))[0]
    pr = pr.reshape((output_height,  output_width, n_classes)).argmax(axis=2)
    

    if out_fname is not None:
        seg_img = visualize_segmentation(pr, inp, n_classes=n_classes,
                                      colors=colors, overlay_img=overlay_img,
                                      show_legends=show_legends,
                                      class_names=class_names,
                                      prediction_width=prediction_width,
                                      prediction_height=prediction_height)
        cv2.imwrite(out_fname, seg_img)
    
    return pr


def predict_multiple(model=None, inps=None, inp_dir=None, out_dir=None,
                     overlay_img=False,class_names=None, show_legends=False, 
                     colors=class_colors,prediction_width=None, prediction_height=None):


    if inps is None and (inp_dir is not None):
        inps = glob.glob(os.path.join(inp_dir, "*.jpg")) + glob.glob(
            os.path.join(inp_dir, "*.png")) + \
            glob.glob(os.path.join(inp_dir, "*.jpeg"))
        inps = sorted(inps)

    assert type(inps) is list

    all_prs = []

    for i, inp in enumerate(tqdm(inps)):
        if out_dir is None:
            out_fname = None
        else:
            if isinstance(inp, six.string_types):
                out_fname = os.path.join(out_dir, os.path.basename(inp))
            else:
                out_fname = os.path.join(out_dir, str(i) + ".jpg")

        pr = predict(model, inp, out_fname,
                     overlay_img=overlay_img, class_names=class_names,
                     show_legends=show_legends, colors=colors,
                     prediction_width=prediction_width,
                     prediction_height=prediction_height)

        all_prs.append(pr)

    return all_prs


def evaluate(model=None, inp_images=None, annotations=None,
             inp_images_dir=None, annotations_dir=None,pre_process=True):


    if inp_images is None:
        assert (inp_images_dir is not None),\
                "Please provide inp_images or inp_images_dir"
        assert (annotations_dir is not None),\
            "Please provide inp_images or inp_images_dir"

        paths = get_pairs_from_paths(images_path=inp_images_dir, segs_path=annotations_dir)
        paths = list(zip(*paths))
        inp_images = list(paths[0])
        annotations = list(paths[1])

    assert type(inp_images) is list
    assert type(annotations) is list

    tp = np.zeros(model.n_classes)
    fp = np.zeros(model.n_classes)
    fn = np.zeros(model.n_classes)
    tn = np.zeros(model.n_classes)
    
    n_pixels = np.zeros(model.n_classes)

    for inp, ann in tqdm(zip(inp_images, annotations)):
        pr = predict(model, inp,pre_process=pre_process)
        
        
        gt = get_segmentation_array(ann, model.n_classes,
                                    model.output_width, model.output_height,
                                    no_reshape=True)
        
        gt = gt.argmax(-1)
        pr = pr.flatten()
        gt = gt.flatten()
        
        for cl_i in range(model.n_classes):

            tp[cl_i] += np.sum((pr == cl_i) * (gt == cl_i))
            fp[cl_i] += np.sum((pr == cl_i) * ((gt != cl_i)))
            fn[cl_i] += np.sum((pr != cl_i) * ((gt == cl_i)))
            tn[cl_i] += np.sum((pr != cl_i) * ((gt != cl_i)))
            n_pixels[cl_i] += np.sum(gt == cl_i)
        
    accuracy=np.sum(tp)/np.sum(n_pixels)
    cl_wise_precision=tp/(tp+fp)
    cl_wise_recall=tp/(tp+fn)
    cl_wise_Fscore=2*(cl_wise_precision*cl_wise_precision)/(cl_wise_precision+cl_wise_precision)
    cl_wise_score = tp / (tp + fp + fn + 0.000000000001)
    n_pixels_norm = n_pixels / np.sum(n_pixels)
    frequency_weighted_IU = np.sum(cl_wise_score*n_pixels_norm)
    mean_IU = np.mean(cl_wise_score)

    return {
        "accuracy": accuracy,
        "frequency_weighted_IOU": frequency_weighted_IU,
        "mean_IOU": mean_IU,
        "class_wise_IOU": cl_wise_score,
        "class_wise_precision": cl_wise_precision,
        "class_wise_recall": cl_wise_recall,
        "class_wise_Fscore": cl_wise_Fscore
    }     


def measure_segmentation_metrics(pr_list,gt_list):
    tp = np.zeros(model_cell.n_classes)
    fp = np.zeros(model_cell.n_classes)
    fn = np.zeros(model_cell.n_classes)
    tn = np.zeros(model_cell.n_classes)
    n_pixels = np.zeros(model_cell.n_classes)
    for pr,gt in zip(pr_list,gt_list):
        
        pr = pr.flatten()
        gt = gt.flatten()
        for cl_i in range(model_cell.n_classes):
    
                tp[cl_i] += np.sum((pr == cl_i) * (gt == cl_i))
                fp[cl_i] += np.sum((pr == cl_i) * ((gt != cl_i)))
                fn[cl_i] += np.sum((pr != cl_i) * ((gt == cl_i)))
                tn[cl_i] += np.sum((pr != cl_i) * ((gt != cl_i)))
                n_pixels[cl_i] += np.sum(gt == cl_i)
        
    accuracy=np.sum(tp)/np.sum(n_pixels)
    cl_wise_precision=tp/(tp+fp)
    cl_wise_recall=tp/(tp+fn)
    cl_wise_Fscore=2*(cl_wise_precision*cl_wise_recall)/(cl_wise_precision+cl_wise_recall)
    cl_wise_score = tp / (tp + fp + fn + 0.000000000001)
    n_pixels_norm = n_pixels / np.sum(n_pixels)
    frequency_weighted_IU = np.sum(cl_wise_score*n_pixels_norm)
    mean_IU = np.mean(cl_wise_score)
    
    results={
        "accuracy": accuracy,
        "frequency_weighted_IOU": frequency_weighted_IU,
        "mean_IOU": mean_IU,
        "class_wise_IOU": cl_wise_score,
        "class_wise_precision": cl_wise_precision,
        "class_wise_recall": cl_wise_recall,
        "class_wise_Fscore": cl_wise_Fscore
    }
    return results

def measure_detection_metrics(pr_labels_list,gt_labels_list):
    
    tp,fp,fn=0,0,0
    for pr_lable_list,gt_label_list in zip(pr_labels_list,gt_labels_list):
        pr_labels,pr_num_labels = pr_lable_list
        gt_labels, gt_num_labels = gt_label_list
        tp_curr,fp_curr,fn_curr=0,0,0
        for pr_label_id in range(1,pr_num_labels+1):
            mask1=(pr_labels==pr_label_id).astype('float64')
            gt_label_ids=np.unique(mask1*gt_labels)
            gt_label_ids=gt_label_ids[gt_label_ids!=0]
            det_flag=0
            for gt_label_id in gt_label_ids:
                mask2=(gt_labels==gt_label_id)
                IOU=np.sum(mask1*mask2)/(np.sum(mask1+mask2)-np.sum(mask1*mask2))
                if IOU>0.7:
                    tp_curr+=1
                    det_flag=1
                    break
            if det_flag==0:
                fp_curr+=1
        
        fn+=(gt_num_labels-tp_curr)
        tp+=tp_curr
        fp+=fp_curr 

    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    Fscore=2*(precision*recall)/(precision+recall)
    
    results={
        "precision": precision,
        "recall": recall,
        "Fscore": Fscore
    }
    return results

def detect_and_track_cells(model_cell,model_tracker,inp_images_dir=None,save_path=None,pad_size=15):

    
    assert (inp_images_dir is not None),\
            "Please provide inp_images_dir"
        

    inp_images = get_pairs_from_paths(images_path=inp_images_dir, ignore_matching=True)
    
    assert type(inp_images) is list
   
    cell_labels=[]
    cell_centroids=[]
    tracked_images=[]
    for idx,inp in tqdm(enumerate(inp_images)):
        
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
    
        cv2.imwrite(os.path.join(save_path, inp.split('/')[-1]), add_labels_to_image(inp,labels_curr,cell_centroids[-1]))