from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import torch

class WCamDataset(Dataset):
# implementation of the dataloading functionalities for 
# training and testing of WCamNet
    def __init__(self, data_path, transform):
        # data provided in csv-format (image path, friction value)
        self.data = pd.read_csv(data_path, header=None)
        self.data_len = len(self.data.index)
        # assume a GPU is available
        self.device = torch.device("cuda")
        # create arrays of image paths and friction values
        self.img_path = np.asarray(self.data.iloc[:, 1])
        self.friction = np.asarray(self.data.iloc[:, 6])
        # transforms are used for data augmentation and proper formatting of input data
        self.transform = transform
    
    def __getitem__(self, index):
        # load image and corresponding friction value
        img_path = self.img_path[index]
        img = Image.open(img_path).convert("RGB")
        friction = self.friction[index]
        # apply image transformations
        img = self.transform(img)
        # ensure friction is provided as 32-bit
        friction = np.float32(friction)
        return (img, friction)

    def __len__(self):
        return self.data_len
