from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# ************* Feature extraction *************

##### https://github.com/lukemelas/EfficientNet-PyTorch#example-classification
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from PIL import Image, ImageOps
import os
import argparse
import time
import os, sys
from time import strftime
from sklearn.metrics import mean_squared_error, accuracy_score, hamming_loss, roc_curve, auc, f1_score
import copy
from torch.utils.data import DataLoader, Dataset
import random
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import PIL.Image as Image
import torch.optim as optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import cv2
from tensorflow.keras.applications import EfficientNetB0
import tensorflow as tf #2.8.3
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence

def load_data_folder(classn, folder, is_train, mask_path = ''): # only load the image filename and the labels
    img_pos = []
    img_neg = []
    lines = [line.rstrip('\n') for line in open(folder + '/labels.txt')]
    for line in lines:
        img = line.split()[0]
        # change the label threshold to generate labels
        if int(line.split()[1]) < -1: continue

        lab = np.array([int(int(line.split()[1]) > 0)])       # class lymphocyte
        
        img_file = folder + '/' + img
        if not os.path.isfile(img_file):
            print('file not exist: ', img_file)
            continue

        if lab > 0:
            #img_pos.append(np.array([img_file, lab]))
            img_pos.append(img_file)
        else:
            #img_neg.append(np.array([img_file, np.array(lab)]))
            img_neg.append(img_file)
    return img_pos, img_neg

def load_data_split(classn, folders, is_train, mask_path = ''):
    X_pos = []
    X_neg = []
    for folder in folders:
        img_pos, img_neg = load_data_folder(classn, folder, is_train, mask_path = '')
        X_pos += img_pos
        X_neg += img_neg
    return X_pos, X_neg

def shuffle_data(data, N_limit = 1): # data is a list
    rands = np.random.permutation(len(data))
    out = []
    count = 0
    if N_limit == 1: N_limit = len(data)
    for i in rands:
        out.append(data[i])
        count += 1
        if count == N_limit:
            break
    return out

def load_imgs_files(classn = 1, dataset_list = '', training_data_path = '', mask_path = ''):
    img_test_pos = []
    img_test_neg = []
    img_train_pos = []
    img_train_neg = []
    lines = [line.rstrip('\n') for line in open(dataset_list)]
    valid_i = 0
    for line in lines:
        split_folders = [training_data_path + "/" + s for s in line.split()]
        if valid_i == 0:
            X_pos, X_neg = load_data_split(classn, split_folders, False,mask_path = '')
            img_test_pos += X_pos
            img_test_neg += X_neg
        else:
            X_pos, X_neg = load_data_split(classn, split_folders, False,mask_path = '')
            img_train_pos += X_pos
            img_train_neg += X_neg
        valid_i += 1

    img_train = img_train_pos + img_train_neg
 
    img_val = img_test_pos + img_test_neg
    img_val = shuffle_data(img_val)

    print("training set loaded, pos: {}; neg: {}".format(len(img_train_pos), len(img_train_neg)))
    print("val set, pos: {}; neg: {}".format(len(img_test_pos), len(img_test_neg)))
    #return np.asarray(img_train), np.asarray(img_val)
    return img_train_pos, img_train_neg, img_test_pos, img_test_neg

list_data = r"C:/Users/emm75/Documents/BMI_771/BMI771_Project/cancer_training/data/training_data/data_files/Val_sm.txt"
data_folder = r"C:/Users/emm75/Documents/BMI_771/BMI771_Project/cancer_training/data/training_data/data_files/data"

img_train_pos, img_train_neg, img_val_pos, img_val_neg = load_imgs_files(2, list_data, data_folder)
# img_train_pos, img_train_neg, img_test_pos, img_test_neg
print(len(img_train_neg))
# for i in img_train_pos:
# for i in img_train_neg:
# for i in img_val_pos:
for i in img_val_neg:
    print(i)
    transform = transforms.CenterCrop(224)
    img = transform(Image.open(i))
    p = os.path.basename(i)
    img.save(p)

# "C:\Users\emm75\Documents\BMI_771\BMI771_Project\crops"