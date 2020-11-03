from keras.models import *
from keras.layers import *
from .deepsea_detector_utils import res_block,get_segmentation_model,add_batch_norm



def deepsea_detector(n_classes=2,input_size = (512,512,1),init_depth=64,dropout=0.2,activation='relu'):
    img_input = Input(shape=input_size)
    
    conv1=res_block(img_input,init_depth,dropout)
    conv1 =add_batch_norm(conv1)
    pool1 = MaxPooling2D((2, 2) )(conv1)
    
    conv2=res_block(pool1,init_depth*2,dropout)
    conv2 =add_batch_norm(conv2)
    pool2 = MaxPooling2D((2, 2) )(conv2)
    
    conv3=res_block(pool2,init_depth*4,dropout)
    conv3 =add_batch_norm(conv3)
    
    up1 = concatenate([UpSampling2D((2, 2) )(conv3), conv2], axis=3)
    conv4=res_block(up1,init_depth*2,dropout)
    conv4 =add_batch_norm(conv4)
    
    up2 = concatenate([UpSampling2D((2, 2) )(conv4), conv1], axis=3)
    conv5=res_block(up2,init_depth,dropout)

    conv6 = Conv2D(init_depth, (3, 3),padding='same')(conv5)
    conv6 =add_batch_norm(conv6)
    
    conv6 = Conv2D(n_classes, (3, 3), activation = activation,padding = 'same')(conv6)
    
    out = Conv2D( n_classes, (1, 1) , padding='same')(conv6)

    model = get_segmentation_model(img_input ,  out ) 
    
    return model