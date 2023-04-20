import torch.utils.data as data
import segmentation_transforms as transforms
import numpy as np
import argparse
import os
import random
from model import DeepSeaSegmentation
from data import BasicSegmentationDataset
import torch
from evaluate import evaluate_segmentation
from utils import get_n_params

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def test(args,image_size = [383,512],image_means = [0.5],image_stds= [0.5],batch_size=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_transforms = transforms.Compose([
                               transforms.ToPILImage(),
                               transforms.Resize(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean = image_means,
                                                    std = image_stds)
                           ])


    test_data = BasicSegmentationDataset(os.path.join(args.test_set_dir, 'images'), os.path.join(args.test_set_dir, 'masks'),os.path.join(args.test_set_dir, 'wmaps'),transforms=test_transforms)

    test_iterator = data.DataLoader(test_data,batch_size = batch_size,shuffle=False)

    model=DeepSeaSegmentation(n_channels=1, n_classes=2, bilinear=True)
    print('INFO: Num of model parameters:',get_n_params(model))
    model.load_state_dict(torch.load(args.ckpt_dir))
    model = model.to(device)

    test_score, test_avg_precision,test_easy_avg_precision,test_hard_avg_precision = evaluate_segmentation(model, test_iterator, device,len(test_data),is_avg_prec=True,prec_thresholds=[0.5,0.6,0.7,0.8,0.9],output_dir=args.output_dir)
    print('INFO: Dice score:', test_score)
    print('INFO: Average precision at ordered thresholds:', test_avg_precision)
    print('INFO: Easy samples average precision at ordered thresholds:', test_easy_avg_precision)
    print('INFO: Hard samples average precision at ordered thresholds:', test_hard_avg_precision)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_set_dir",required=True,type=str,help="path for the test dataset")
    ap.add_argument("--ckpt_dir",required=True,type=str,help="path for the checkpoint of segmentation model to test")
    ap.add_argument("--output_dir", required=True, type=str, help="path for saving the test outputs")

    args = ap.parse_args()

    assert os.path.isdir(args.test_set_dir), 'No such file or directory: ' + args.test_set_dir
    if not os.path.isdir(os.path.join(args.output_dir,'input_segmentation_images')):
        os.makedirs(os.path.join(args.output_dir,'input_segmentation_images'))
    if not os.path.isdir(os.path.join(args.output_dir,'segmentation_predictions')):
        os.makedirs(os.path.join(args.output_dir,'segmentation_predictions'))

    test(args)
