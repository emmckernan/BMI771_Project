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
import math

mask = r"C:\Users\emm75\Documents\BMI_771\BMI771_Project\cancer_training\data\training_data\data_files\TCGA-BH-A0B6-01Z-00-DX1.4D982935-F600-4F37-ABB5-13AF74F4FDC4.png"
img = Image.open(mask)
# DONE: for TCGA-C8-A26V-01Z-00-DX1.6E86DF74-D575-4969-8142-963D9DF5208F : 74796+2137 / 8549+2137
# DONE: for TCGA-A1-A0SE-01Z-00-DX1.04B09232-C6C4-46EF-AA2C-41D078D0A80A : 120757 + 2082 / 72871 + 2082
# DONE: for TCGA-AR-A0TX-01Z-00-DX1.5BEA4E65-8CEC-49B2-9F29-DBE53E8ED46E : x = 70192 + 2127 / y = 65938 + 2127
# DONE: for TCGA-A2-A0CV-01Z-00-DX1.B02D017A-61CD-45FE-BB31-705CD127DDAB : 73062/159380
# DONE: for TCGA-B6-A0RP-01Z-00-DX1.02A55E9D-2AA9-497A-B481-85724EA813AD : 87584/107533
# DONE: for TCGA-E2-A153-01Z-00-DX1.CA994467-E541-4131-A9FC-DCD9944F29C4 : 69021/80920
# DONE: for TCGA-B6-A0X5-01Z-00-DX1.02A9FF1E-EA20-4F2D-A2BB-427A287DE3FD : 62423/123528
# DONE: for TCGA-BH-A0B6-01Z-00-DX1.4D982935-F600-4F37-ABB5-13AF74F4FDC4 : 77884/140765

x = 123528
y = 62423
print(x/2095,y/2095)
pixel = img.load()

folder_data = r"C:/Users/emm75/Documents/BMI_771/BMI771_Project/cancer_training/data/training_data/data_files/TCGA-BH-A0B6-01Z-00-DX1.4D982935-F600-4F37-ABB5-13AF74F4FDC4.svs/"
entries = os.scandir(folder_data)

count = 0

names = []

with os.scandir(folder_data) as entries:
    for entry in entries:
        coor = entry.name.split("_")
        #print(coor)
        s = coor[2].translate(' \n\t\r')
        cx = coor[0].translate(' \n\t\r')
        cy = coor[1].translate(' \n\t\r')
        new_cx = math.ceil(int(cx)/int(s))
        new_cy = math.ceil(int(cy)/int(s))
        #print(cx,cy,new_cx, new_cy)
        dif_x = img.size[0]/59
        dif_y = img.size[1]/30
        x1 = int(new_cx*dif_x-dif_x+1)
        x2 = int(new_cx*dif_x+1)
        y1 = int(new_cy*dif_y-dif_y+1)
        y2 = int(new_cy*dif_y+1)
        #print(x1,x2,y1,y2)
        crop = img.crop((x1,y1,x2,y2))
        pos = '0'
        if np.mean(crop) > 0:
            pos = '1'
        #print(pos)
        n = entry.name
        names.append((n+" "+pos))
        #print(cx,cy,img.getpixel((new_cx*dif_x,new_cy*dif_y)))
        #print(new_cx,new_cy)
        count += 1
        #print(float(coor[0].strip()))
        
# print((img.size[0]),(img.size[1]))

print(names)

with open(("C:/Users/emm75/Documents/BMI_771/BMI771_Project/cancer_training/data/training_data/data_files/TCGA-BH-A0B6-01Z-00-DX1.4D982935-F600-4F37-ABB5-13AF74F4FDC4.svs/labels.txt"), 'w') as f:
    f.write('\n'.join(names))
