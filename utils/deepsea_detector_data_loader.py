import itertools
import os
import random
import six
import cv2
import numpy as np
from PIL import Image
try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found, disabling progress bars")

    def tqdm(iter):
        return iter

DATA_LOADER_SEED = 9

random.seed(DATA_LOADER_SEED)
class_colors = [(random.randint(0, 255), random.randint(
    0, 255), random.randint(0, 255)) for _ in range(5000)]

class DataLoaderError(Exception):
    pass

def get_pairs_from_paths(images_path, segs_path=None, ignore_matching=False):
    """ Find all the images from the images_path directory and
        the segmentation images from the segs_path directory and
        the weight maps from the weights_path directory
        while checking integrity of data """

    ACCEPTABLE_IMAGE_FORMATS = [".tif",".jpg", ".jpeg", ".png", ".bmp"]
    ACCEPTABLE_SEGMENTATION_FORMATS = [".tif",".png", ".bmp"]
    
    image_files = []
    segmentation_files = {}


    for dir_entry in os.listdir(images_path):
        if os.path.isfile(os.path.join(images_path, dir_entry)) and \
                os.path.splitext(dir_entry)[1] in ACCEPTABLE_IMAGE_FORMATS:
            file_name, file_extension = os.path.splitext(dir_entry)
            image_files.append((file_name, file_extension,
                                os.path.join(images_path, dir_entry)))
    if segs_path:
        for dir_entry in os.listdir(segs_path):
            if os.path.isfile(os.path.join(segs_path, dir_entry)) and \
               os.path.splitext(dir_entry)[1] in ACCEPTABLE_SEGMENTATION_FORMATS:
                file_name, file_extension = os.path.splitext(dir_entry)
                full_dir_entry = os.path.join(segs_path, dir_entry)
                if file_name in segmentation_files:
                    raise DataLoaderError("Segmentation file with filename {0}"
                                          " already exists and is ambiguous to"
                                          " resolve with path {1}."
                                          " Please remove or rename the latter."
                                          .format(file_name, full_dir_entry))
    
                segmentation_files[file_name] = (file_extension, full_dir_entry)
    
    
    return_value = []
    # Match the images and segmentations
    for image_file, _, image_full_path in image_files:
        
        
        if ignore_matching:
            return_value.append(image_full_path)
        elif image_file in segmentation_files:
            return_value.append((image_full_path,
                                segmentation_files[image_file][1]))
        else:
            # Error out
            raise DataLoaderError("No corresponding segmentation "
                                  "found for image {0}."
                                  .format(image_full_path))

    return return_value

def do_pre_process(img,width=512,height=512):
        img=(img-np.min(img))/(np.max(img)-np.min(img))
        clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8))
        img = clahe.apply((img*255).astype('uint8'))
        img=(img-np.min(img))/(np.max(img)-np.min(img))
        img=cv2.resize(img, (width, height))
        return img        
            
def get_image_array(image_input,
                    width=512, height=512,reshape=False,pre_process=True):
    """ Load image array from input """

    if type(image_input) is np.ndarray:
        # It is already an array, use it as it is
        img = image_input
    elif isinstance(image_input, six.string_types):
        if not os.path.isfile(image_input):
            raise DataLoaderError("get_image_array: path {0} doesn't exist"
                                  .format(image_input))
        img = Image.open(image_input)
        img=np.array(img)
        
    else:
        raise DataLoaderError("get_image_array: Can't process input type {0}"
                              .format(str(type(image_input))))
        
    img = img.astype(np.float64)
    if len(img.shape)==3:
        img=img[:, :, 0]
    if pre_process:
        img=do_pre_process(img,width, height)
    else:
        img = cv2.resize(img, (width, height))
        img = img/255.0
    if reshape:    
        img=img.reshape((width, height,1))

    return img


def get_segmentation_array(image_input, nClasses,width, height, no_reshape=True):
    """ Load segmentation array from input """

    seg_labels = np.zeros((height, width, nClasses))

    if type(image_input) is np.ndarray:
        # It is already an array, use it as it is
        img = image_input
    elif isinstance(image_input, six.string_types):
        if not os.path.isfile(image_input):
            raise DataLoaderError("get_segmentation_array: "
                                  "path {0} doesn't exist".format(image_input))
        
        img = Image.open(image_input)
        img=np.array(img)
    else:
        raise DataLoaderError("get_segmentation_array: "
                              "Can't process input type {0}"
                              .format(str(type(image_input))))

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)

    if len(img.shape)==3:
        img = img[:, :, 0]
   

    for c in range(nClasses):
        seg_labels[:, :, c] = (img == c).astype(int)
    
        
    if not no_reshape:
        seg_labels = np.reshape(seg_labels, (width*height, nClasses))

    return seg_labels


def verify_segmentation_dataset(images_path, segs_path,
                                n_classes, show_all_errors=False):
    try:
        img_seg_pairs = get_pairs_from_paths(images_path, segs_path)
        if not len(img_seg_pairs):
            print("Couldn't load any data from images_path: "
                  "{0} and segmentations path: {1}"
                  .format(images_path, segs_path))
            return False

        return_value = True
        for im_fn, seg_fn  in tqdm(img_seg_pairs):
            img = Image.open(im_fn)
            img=np.array(img)
            if len(img.shape)==3:
                img=img[:, :, 0]
            seg = Image.open(seg_fn)
            seg=np.array(seg)
            if len(seg.shape)==3:
                seg=seg[:, :, 0]
            

            # Check dimensions match
            if not (img.shape == seg.shape) :
                return_value = False
                print("The size of image {0} and its segmentation {1} "
                      "doesn't match (possibly the files are corrupt)."
                      .format(im_fn, seg_fn))
                if not show_all_errors:
                    break
           
            else:
                max_pixel_value = np.max(seg)
                if max_pixel_value >= n_classes:
                    return_value = False
                    print("The pixel values of the segmentation image {0} "
                          "violating range [0, {1}]. "
                          "Found maximum pixel value {2}"
                          .format(seg_fn, str(n_classes - 1), max_pixel_value))
                    if not show_all_errors:
                        break
        if return_value:
            print("Dataset verified! ")
        else:
            print("Dataset not verified!")
        return return_value
    except DataLoaderError as e:
        print("Found error during data loading\n{0}".format(str(e)))
        return False


def image_segmentation_generator(images_path, segs_path,batch_size,
                                 n_classes, input_height, input_width,
                                 output_height, output_width):

    img_seg_pairs = get_pairs_from_paths(images_path, segs_path)
    random.shuffle(img_seg_pairs)
    zipped = itertools.cycle(img_seg_pairs)

    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            im, seg = next(zipped)
            im = Image.open(im)
            im=np.array(im)
            if len(im.shape)==3:
                im=im[:, :, 0]
            
            seg = Image.open(seg)
            seg=np.array(seg)
            if len(seg.shape)==3:
                seg=seg[:, :, 0]
        
            X.append(get_image_array(im, input_width,
                                     input_height, reshape=True,pre_process=False))
            Y.append(get_segmentation_array(
                seg,n_classes, output_width, output_height))

        yield np.array(X), np.array(Y)
