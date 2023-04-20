import torchvision.transforms as transforms
from skimage.morphology import remove_small_objects
import numpy as np
import argparse
import cv2
import os
from scipy import ndimage as ndi
import random
from model import DeepSeaSegmentation
import torch
from utils import visualize_segmentation


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def apply_img_segmentation(model_ckpt,img,image_size = [383,512],image_means = [0.5],image_stds= [0.5]):
    """ function to apply single shot segmentation

        Parameters
        ------------

        model_ckpt: Pre-trained segmentation model checkpoint
        img: Input image (CV2 Numpy Image)

        Returns
        ------------

        binary_mask: binary mask of segmented cells (numpy array)
        overlay_img: overlay image of segmented cells (numpy array), each color represent a single cell

    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_transforms = transforms.Compose([
                               transforms.ToPILImage(),
                               transforms.Resize(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean = image_means,
                                                    std = image_stds)
                           ])
    img = (255 * ((img - img.min()) / img.ptp())).astype(np.uint8)
    tensor_img=test_transforms(img).to(device=device, dtype=torch.float32)
    model=DeepSeaSegmentation(n_channels=1, n_classes=2, bilinear=True)
    model.load_state_dict(model_ckpt)
    model = model.to(device)
    model=model.eval()
    mask_pred, edge_pred = model(tensor_img.unsqueeze(0))
    mask_pred = mask_pred.argmax(dim=1).cpu().numpy()
    label_img, _ = ndi.label(remove_small_objects(mask_pred[0, :, :] > 0, min_size=20, connectivity=1))
    img=tensor_img.cpu().numpy()[0,:,:]
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
    overlay_img = visualize_segmentation(label_img, inp_img=img, overlay_img=True)
    binary_mask=label_img.copy()
    binary_mask[binary_mask>0]=255
    return label_img,binary_mask,overlay_img,img


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--single_img_dir",required=True,type=str,help="path for the single test image")
    ap.add_argument("--ckpt_dir",required=True,type=str,help="path for the checkpoint of segmentation model to test")
    ap.add_argument("--output_dir", required=True, type=str, help="path for saving the test outputs")

    args = ap.parse_args()
    assert os.path.isfile(args.single_img_dir), 'No such file or directory: ' + args.single_img_dir
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    model_ckpt=torch.load(args.ckpt_dir)
    img = cv2.imread(args.single_img_dir,0)
    label_img,binary_mask,overlay_img,img=apply_img_segmentation(model_ckpt,img)
    cv2.imwrite(os.path.join(args.output_dir, 'label_img.png'), label_img)
    cv2.imwrite(os.path.join(args.output_dir, 'binary_mask.png'), binary_mask)
    cv2.imwrite(os.path.join(args.output_dir, 'overlay_img.png'), overlay_img)
    cv2.imwrite(os.path.join(args.output_dir, 'original_img_resized.png'), img)