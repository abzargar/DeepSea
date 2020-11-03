from keras.models import *
from keras.layers import *
from .deepsea_tracker_utils import res_block,get_segmentation_model,add_batch_norm


def deepsea_tracker(n_classes=2,input_size = (128,128,1),init_depth=64,dropout=0.2,activation='relu'):

    input_curr = Input(shape=input_size)
    input_prev = Input(shape=input_size)
    
    conv1_1=res_block(input_curr,init_depth)
    conv1_1 =add_batch_norm(conv1_1)
    # conv1_1=res_block(conv1_1,init_depth)
    # conv1_1 =add_batch_norm(conv1_1)
    
    conv1_2=res_block(input_prev,init_depth)
    conv1_2 =add_batch_norm(conv1_2)
    # conv1_2=res_block(conv1_2,init_depth)
    # conv1_2 =add_batch_norm(conv1_2)
    
    conv1=concatenate([conv1_1,conv1_2], axis = 3)
    
    
    pool1 = MaxPooling2D((2, 2) )(conv1)
    
    conv2 = res_block(pool1,init_depth*2)
    conv2 =add_batch_norm(conv2)
    
    pool2 = MaxPooling2D((2, 2) )(conv2)
    
    conv3 = res_block(pool2,init_depth*4)
    conv3 =add_batch_norm(conv3)
    
    up1 = concatenate([UpSampling2D((2, 2) )(conv3), conv2], axis=3)
    conv4=res_block(up1,init_depth*2)
    conv4 =add_batch_norm(conv4)
    
    up2 = concatenate([UpSampling2D((2, 2) )(conv4), conv1], axis=3)
    conv5=res_block(up2,init_depth,dropout)
   
    
    conv6 = Conv2D(init_depth, (3, 3),padding='same')(conv5)
    conv6 =add_batch_norm(conv6)
    
    conv6 = Conv2D(n_classes, (3, 3), activation = activation,padding = 'same')(conv6)
    
    out = Conv2D( n_classes, (1, 1) , padding='same')(conv6)

   
    model = get_segmentation_model([input_curr, input_prev] ,  out  ) 
    
    return model
