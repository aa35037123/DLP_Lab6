import pandas as pd
from torch.utils import data
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
import json
from sklearn import preprocessing
from torch.optim.lr_scheduler import StepLR
import torch
import os
import numpy as np
import itertools
from torchvision.utils import save_image
import torchvision

def getCode():
    path = './' + 'objects.json'
    with open(path) as file:
        code = json.load(file)
    
    return code
def getTrainData(mode, code):
    path = './' + mode + '.json'
    with open(path) as file:
        data = json.load(file)
    # LabelBinarizer transform classify label into one-hot vector 
    lb = preprocessing.LabelBinarizer() # this is pretrain classifier
    lb.fit([i for i in range(24)]) # there are total 23 different object in object.json
    
    img_name = []
    labels = []
    
    for name, shapes in data.items():
        img_name.append(name)
        tmp = []
        for shape in shapes:
            tmp.append(np.array(lb.transform([code[shape]]))) # LabelBinarizer transform classify label into one-hot vector 
        labels.append((np.sum(tmp, axis=0))) # sum up by row
    # print('train_img_name:', len(img_name))
    # print("train_labels:", len(labels))
    labels = torch.tensor(np.array(labels))
    
    return img_name, labels
    
def getTestData(mode, code):
    path = './' + mode + '.json'
    with open(path) as file:
        data = json.load(file)
    lb = preprocessing.LabelBinarizer()
    lb.fit([i for i in range(24)])
    labels = []
    for shapes in data:
        tmp = []
        for shape in shapes:
            tmp.append(np.array(lb.transform([code[shape]]))) # transform into one-hot vector
        labels.append(np.sum(tmp, axis=0))
    print("test_labels:", len(labels))
    labels = torch.tensor(np.array(labels))
    
    return labels
class iclevrLoader(data.Dataset):
    def __init__(self, args, data_path, mode):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.args = args
        self.data_path = data_path
        self.code = getCode() # data means (k. v) pair : (shape, value) in objects.json 
        self.mode = mode
        # get datas
        if self.mode == 'train':
            self.image_name, self.label = getTrainData(self.mode, self.code)
        elif self.mode == 'test' or self.mode == 'new_test':
            self.label = getTestData(self.mode, self.code)
        else:
            raise ValueError('No such root!')
        
    def __len__(self):
        """'return the size of dataset"""
        return len(self.label)
    def __getitem__(self, index):
        # Return processed image and label
        if self.mode == 'train':
            path = os.path.join(self.data_path, self.image_name[index])
            transform_img = transforms.Compose([
                transforms.Resize([64, 64]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # normalize is important, in order to put pixel value into [-1, 1]
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
            ])
            img = transform_img(Image.open(path).convert('RGB'))
        else:
            img = np.random.randn(3, 64, 64) # in test stage, we don't need image value, it just a dummy value
        
        label = self.label[index]
        return img, label

def save_images(args, images, name):
    save_image(images, fp=os.path.join(args.save_root, f'{name}.png'))
    
# l = iclevrLoader("dataset/", "test")