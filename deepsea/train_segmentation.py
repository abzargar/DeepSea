import argparse
from model import DeepSeaSegmentation
from data import BasicSegmentationDataset
import torch.nn as nn
from evaluate import evaluate_segmentation
from loss import dice_loss
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torch.nn.functional as F
import segmentation_transforms as transforms
import torch
import numpy as np
import os
import random
from tqdm import tqdm
import logging

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def train(args,image_size = [383,512],image_means = [0.5],image_stds= [0.5],train_ratio = 0.85,save_checkpoint=True,if_train_aug=True,train_aug_iter=3,patience=5):
    """ function to train the segmentation model

            Parameters
            ------------

            image_size: used to  resize the input image to the given size
            image_means: used for the input image normalization
            image_stds: used for the input image normalization
            train_ratio: ratio of training samples to train the model, the remaining go for the validation process
            save_checkpoint: True/False, True saves the best checkpoint given the validation score during the training process
            if_train_aug: True/False, True uses the image augmentation during the training process (Recommended)
            train_aug_iter: Number of augmentation iterations requested for each original training image
            patience: number of epochs with no improvement before it stops the training, used for the early stopping technique


    """
    logging.basicConfig(filename=os.path.join(args.output_dir, 'train.log'), filemode='w',format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info('>>>> image size=(%d,%d) , learning rate=%f , batch size=%d' % (image_size[0], image_size[1],args.lr,args.batch_size))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if if_train_aug:
        train_transforms = transforms.Compose([
                                   transforms.ToPILImage(),
                                   transforms.RandomApply([transforms.RandomOrder([
                                       transforms.RandomApply([transforms.ColorJitter(brightness=0.33, contrast=0.33, saturation=0.33, hue=0)],p=0.5),
                                       transforms.RandomApply([transforms.GaussianBlur((5, 5), sigma=(0.1, 1.0))],p=0.5),
                                       transforms.RandomApply([transforms.RandomHorizontalFlip(0.5)],p=0.5),
                                       transforms.RandomApply([transforms.RandomVerticalFlip(0.5)],p=0.5),
                                       transforms.RandomApply([transforms.AddGaussianNoise(0., 0.05)], p=0.5),
                                       transforms.RandomApply([transforms.CLAHE()], p=0.5),
                                       transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=2)], p=0.5),
                                       transforms.RandomApply([transforms.RandomCrop()], p=0.5),
                                    ])],p=1-1/(train_aug_iter+1)),
                                   transforms.Resize(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean = image_means,std = image_stds)
                               ])
    else:
        train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=image_means,
                                 std=image_stds)
        ])


    train_data = BasicSegmentationDataset(os.path.join(args.train_set_dir, 'images'), os.path.join(args.train_set_dir, 'masks'),os.path.join(args.train_set_dir, 'wmaps'),transforms=train_transforms,if_train_aug=if_train_aug,train_aug_iter=train_aug_iter)

    n_train_examples = int(len(train_data) * train_ratio)
    n_valid_examples = len(train_data) - n_train_examples

    train_data, valid_data = data.random_split(train_data,[n_train_examples, n_valid_examples],generator=torch.Generator().manual_seed(SEED))

    train_iterator = data.DataLoader(train_data,shuffle = True,batch_size = args.batch_size)

    valid_iterator = data.DataLoader(valid_data,batch_size = args.batch_size)

    model=DeepSeaSegmentation(n_channels=1, n_classes=2, bilinear=True)
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)

    STEPS_PER_EPOCH = len(train_iterator)
    TOTAL_STEPS = args.max_epoch * STEPS_PER_EPOCH
    MAX_LRS = [p['lr'] for p in optimizer.param_groups]
    scheduler = lr_scheduler.OneCycleLR(optimizer,max_lr=MAX_LRS,total_steps=TOTAL_STEPS)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    criterion = criterion.to(device)
    nstop=0
    avg_precision_best=0
    logging.info('>>>> Start training')
    print('INFO: Start training ...')
    for epoch in range(args.max_epoch):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train_examples, desc=f'Epoch {epoch + 1}/{args.max_epoch}', unit='img') as pbar:
            for step,batch in enumerate(train_iterator):
                images = batch['image']
                true_masks = batch['mask']
                true_wmaps = batch['wmap']
                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                true_wmaps = true_wmaps.to(device=device, dtype=torch.long)
                true_masks=torch.squeeze(true_masks, dim=1)
                true_wmaps = torch.squeeze(true_wmaps, dim=1)
                with torch.cuda.amp.autocast(enabled=True):
                    masks_preds,wmap_preds = model(images)
                    loss = criterion(masks_preds, true_masks) \
                            +5*criterion(wmap_preds, true_wmaps) \
                            + dice_loss(F.softmax(masks_preds, dim=1).float(),
                                       F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                                       multiclass=True)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                pbar.update(images.shape[0])
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss': epoch_loss/(step+1)})

        # Evaluation round
        val_score,avg_precision,_,_ = evaluate_segmentation(model, valid_iterator, device,n_valid_examples,is_avg_prec=((1+epoch)%2==0),prec_thresholds=[0.5])
        if avg_precision is not None:
            logging.info('>>>> Epoch:%d  , loss=%f , valid score=%f , avg precision=%f' % (
            epoch, epoch_loss / (step+1), val_score, avg_precision[0]))
        else:
            logging.info('>>>> Epoch:%d  , loss=%f , valid score=%f' % (
                epoch, epoch_loss / (step + 1), val_score))

        ## Save best checkpoint corresponding the best average precision
        if avg_precision is not None and avg_precision>avg_precision_best:
            avg_precision_best=avg_precision
            states = model.state_dict()
            if save_checkpoint:
               logging.info('>>>> Save the model checkpoint to %s'%(os.path.join(args.output_dir,'segmentation.pth')))
               torch.save(states, os.path.join(args.output_dir,'segmentation.pth'))
            nstop=0
        elif avg_precision is not None and avg_precision<=avg_precision_best:
            nstop+=1
        if nstop==patience:#Early Stopping
            print('INFO: Early Stopping met ...')
            print('INFO: Finish training process')
            break
        scheduler.step()




if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_set_dir",required=True,type=str,help="path for the train dataset")
    ap.add_argument("--lr", default=1e-3,type=float, help="learning rate")
    ap.add_argument("--max_epoch", default=200, type=int, help="maximum epoch to train model")
    ap.add_argument("--batch_size", default=16, type=int, help="train batch size")
    ap.add_argument("--output_dir", required=True, type=str, help="path for saving the train log and best checkpoint")

    args = ap.parse_args()
    assert os.path.isdir(args.train_set_dir), 'No such file or directory: ' + args.train_set_dir
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    train(args)