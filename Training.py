# ************* Feature extraction *************

##### https://github.com/lukemelas/EfficientNet-PyTorch#example-classification
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from PIL import Image
import os

# data_folder = r"./data/TCGA-A2-A25B-01Z-00-DX1.58D7BEDE-5558-4A9E-A95E-DDF24C9267EF.svs"

#for file in os.listdir(data_folder):

# model = EfficientNet.from_pretrained('efficientnet-b0')

# img_path = data_folder + "/4255_48922_2127_2100.png"

# tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
# img = tfms(Image.open(img_path).convert('RGB')).unsqueeze(0)

# # ... image preprocessing as in the classification example ...
# print(img.shape) # torch.Size([1, 3, 224, 224])

# features = model.extract_features(img)
# print(features.shape) # torch.Size([1, 1280, 7, 7])

# #************* Training *************

##pre-trained breast cancer model from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5967747/

import argparse
from torchvision import transforms
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
import os
import sys
import torch.nn as nn
import cv2

from tensorflow.keras.applications import EfficientNetB0
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

def load_data_folder(classn, folder, is_train, mask_path = ''): # only load the image filename and the labels
    img_pos = []
    img_neg = []
    lines = [line.rstrip('\n') for line in open(folder + '/labels.txt')]
    for line in lines:
        #print("......... line:",line)
        img = line.split()[0]
        # change the label threshold to generate labels
        if int(line.split()[1]) < -1: continue

        lab = np.array([int(int(line.split()[1]) > 0)])       # class lymphocyte
        
        # PC_058_0_1-17005-12805-2400-10X-0-macenko.png
        # if color != 'none':
        #     img = img.split('.png')[0] + '_' + color + '.png'

        # # check is the segmentation mask available:
        # if mask_path != '':
        #     seg_file = os.path.join(folder, img.split('.png')[0] + '_reinhard_segment.png')
        #     if not os.path.exists(seg_file):
        #         print('file not exist: ', seg_file)
        #         continue

        img_file = folder + '/' + img
        if not os.path.isfile(img_file):
            print('file not exist: ', img_file)
            continue

        if lab > 0:
            img_pos.append((img_file, lab))
        else:
            img_neg.append((img_file, lab))
    print("\timgs:",len(img_pos), len(img_neg))
    return img_pos, img_neg

def load_data_split(classn, folders, is_train, mask_path = ''):
    X_pos = []
    X_neg = []
    ("^^^^^^",dataset_list)
    for folder in folders:
        img_pos, img_neg = load_data_folder(classn, folder, is_train, mask_path = '')
        X_pos += img_pos
        X_neg += img_neg
    print("\tx:",len(X_pos),len(X_neg))
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
    ("%%%%",dataset_list)
    lines = [line.rstrip('\n') for line in open(dataset_list)]
    valid_i = 0
    for line in lines:
        split_folders = [training_data_path + "/" + s for s in line.split()]
        X_pos, X_neg = load_data_split(classn, split_folders, False,mask_path = '')
        img_train_pos += X_pos
        img_train_neg += X_neg

    print("did it work?")
    img_trains = img_train_pos + img_train_neg
    # ==== testing data ====
    img_vals = img_test_pos + img_test_neg
    img_vals = shuffle_data(img_vals)

    print("training set loaded, pos: {}; neg: {}".format(len(img_train_pos), len(img_train_neg)))
    print("val set, pos: {}; neg: {}".format(len(img_test_pos), len(img_test_neg)))
    return img_trains, img_vals

rand_seed = 26700
if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)

use_gpu = torch.cuda.is_available()
print('Using GPU: ', use_gpu)

device = torch.device("cuda:0")

classn = 1
freq_print = 100     # print stats every {} batches

training_data_path = r'C:/Users/emm75/Documents/BMI_771/BMI771_Project/cancer_training/data/training_data/data_files'
dataset_list = os.path.join(training_data_path, 'test_folders.txt')
print(dataset_list)
dataset_list = os.path.join(training_data_path, dataset_list)

print(dataset_list)

###########
print('DataLoader ....')

def mean_std(type = 'none'):
    mean = [0.7238, 0.5716, 0.6779]
    std = [0.1120, 0.1459, 0.1089]
    return mean, std

mean, std = mean_std()

input_size = 224
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(22),
        transforms.CenterCrop(350),
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(),  # simple data augmentation
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]),

    'val': transforms.Compose([
        transforms.CenterCrop(350),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}
print("#####",type(classn), type(dataset_list), type(training_data_path))
img_trains, img_vals = load_imgs_files(classn, dataset_list = dataset_list, training_data_path = training_data_path)



IMG_SIZE = 224
NUM_CLASSES = 3 # NUM_CLASSES = ds_info.features["label"].num_classes

batch_size = 64

ds_train_image =  (np.array(img_trains)[:,0]).tolist()
ds_train_label = (np.array(img_trains)[:,1]).tolist()
ds_test = data_folder

print(sum(ds_train_label))#, "\n\n",ds_train_label)
size = (IMG_SIZE, IMG_SIZE)
ds_train = map(lambda x: tf.image.resize(x, size), ds_train_image)

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    print("Device:", tpu.master())
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError:
    print("Not connected to a TPU runtime. Using CPU/GPU strategy")
    strategy = tf.distribute.MirroredStrategy()

img_augmentation = Sequential(
    [
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)


with strategy.scope():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = img_augmentation(inputs)
    outputs = EfficientNetB0(include_top=True, weights=None, classes=NUM_CLASSES)(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

model.summary()

epochs = 40  # @param {type: "slider", min:10, max:100}
hist = model.fit(ds_train, epochs=epochs, validation_data=ds_test, verbose=2)

