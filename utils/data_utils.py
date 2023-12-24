import logging
import os
import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler, random_split
import subprocess
import re
from utils.custom_dataset import CustomDataset
import random
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np

logger = logging.getLogger(__name__)

class MyRotateTransform(object):
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = np.random.choice(self.angles, p=[0.8, 0.2])
        return F.rotate(x, angle)
    


data_transforms = {

'train': T.Compose([
	T.RandomResizedCrop(size=(224,224), scale=(0.7,1), ratio=(5/4,5/3)),
	T.RandomHorizontalFlip(),
	MyRotateTransform([0, 180]),
	T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),

'val': T.Compose([
	T.RandomResizedCrop(size=(224,224), scale=(1,1), ratio=(5/4,5/3)),
	T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),

'test': T.Compose([
	T.RandomResizedCrop(size=(224,224), scale=(1,1)),
	T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
}



def get_loader(args):


    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    data = CustomDataset(args.data_dir,label_path=None, num_frames = 64, transform = data_transforms["train"], blackbar_check = None)
    try: 
      trainset, testset = random_split(data, [0.8, 0.2], generator=torch.Generator().manual_seed(args.seed))
    except:
      trainset, testset = random_split(data, [935, 233], generator=torch.Generator().manual_seed(args.seed))
    testset.dataset.set_transform(data_transforms["val"])
        

    if trainset is not None:
        train_sampler = RandomSampler(trainset) 
        train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=0,
                              pin_memory=True)
    else:
        train_loader = None
    if testset is not None:
        test_sampler = SequentialSampler(testset)
        test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=0,
                             pin_memory=True)
    else:
        train_loader = None
        
    

    return train_loader, test_loader

def get_blackbar(vid_path):
    CROP_DETECT_LINE = b'w:(\d+)\sh:(\d+)\sx:(\d+)\sy:(\d+)'
    CROP_COORDINATE = b'x1:(\d+)\sx2:(\d+)\sy1:(\d+)\sy2:(\d+)'
    p = subprocess.Popen(["ffmpeg", "-i", vid_path, "-vf", "cropdetect", "-vframes", "2", "-f", "rawvideo", "-y", "/dev/null"]
                    , stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    infos = p.stderr.read()
    crop_coordinate = re.findall(CROP_COORDINATE , infos) #y1,y2,x1,x2
    crop_data = re.findall(CROP_DETECT_LINE , infos) #(width,height,left,top)
    crop_coordinate = crop_coordinate[0]
    if int(crop_coordinate[0].decode('utf8')) == 0 and int(crop_coordinate[2].decode('utf8')) == 0:
        return None
    else:
        output = [int(crop.decode('utf8')) for crop in crop_data[0]] 
    
    return output



