import os
import torch.utils.data as data
import tracker_transforms as transforms
import numpy as np
import argparse
import random
from model import DeepSeaTracker
from data import BasicTrackerDataset
import torch
from evaluate import evaluate_tracker
from utils import get_n_params

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True



def test(args,image_size = [128,128],image_means = [0.5],image_stds= [0.5],batch_size=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_transforms = transforms.Compose([
                               transforms.ToPILImage(),
                               transforms.Resize(image_size),
                               transforms.Grayscale(num_output_channels=1),
                               transforms.ToTensor(),
                               transforms.Normalize(mean = image_means,
                                                    std = image_stds)
                           ])


    test_data = BasicTrackerDataset(os.path.join(args.test_set_dir), transforms=test_transforms,if_test=True)
    test_iterator = data.DataLoader(test_data,batch_size = batch_size)

    model=DeepSeaTracker(n_channels=1, n_classes=2, bilinear=True)
    print('INFO: Num of model parameters:',get_n_params(model))

    model.load_state_dict(torch.load(args.ckpt_dir))
    model = model.to(device)

    test_score, test_avg_precision,test_single_cell_avg_precision,test_mitosis_avg_precision = evaluate_tracker(model, test_iterator, device,len(test_data),is_avg_prec=True,prec_thresholds=[0.2,0.6,0.7,0.8,0.9],output_dir=args.output_dir)

    print('INFO: Dice score:', test_score)
    print('INFO: Average precision:', test_avg_precision)
    print('INFO: Single cells average precision:', test_single_cell_avg_precision)
    print('INFO: Mitosis average precision:', test_mitosis_avg_precision)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_set_dir",required=True,type=str,help="path for the test dataset")
    ap.add_argument("--ckpt_dir",required=True,type=str,help="path for the checkpoint of tracking model to test")
    ap.add_argument("--output_dir", required=True, type=str, help="path for saving the test outputs")

    args = ap.parse_args()

    assert os.path.isdir(args.test_set_dir), 'No such file or directory: ' + args.test_set_dir
    if not os.path.isdir(os.path.join(args.output_dir,'input_crops')):
        os.makedirs(os.path.join(args.output_dir,'input_crops'))
    if not os.path.isdir(os.path.join(args.output_dir,'tracking_predictions')):
        os.makedirs(os.path.join(args.output_dir,'tracking_predictions'))

    test(args)