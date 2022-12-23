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

# path is 
# C:\Users\emm75\Documents\BMI_771\BMI771_Project\lg_tiles TCGA-A2-A0T7-01Z-00-DX1.EA4DF9B5-8D04-4BCC-9ECB-CB8CB8ACBE1C 81052 99959 DONE
# C:\Users\emm75\Documents\BMI_771\BMI771_Project\lg_tiles TCGA-A8-A07E-01Z-00-DX1.AC684481-979A-46F9-91D7-C56CE85992F2 110336 77824 DONE
# C:\Users\emm75\Documents\BMI_771\BMI771_Project\lg_tiles TCGA-BH-A0BO-01Z-00-DX1.1A704471-FEB3-40F9-9838-3E347A18285F 72418 109453 DONE
# C:\Users\emm75\Documents\BMI_771\BMI771_Project\lg_tiles TCGA-BH-A0DE-01Z-00-DX1.64A0340A-8146-48E8-AAF7-4035988B9152 74206 101772 DONE
# C:\Users\emm75\Documents\BMI_771\BMI771_Project\lg_tiles TCGA-BH-A18H-01Z-00-DX1.4EC9108F-04C2-4B28-BD74-97A414C9A536 90946 131189 DONE
# C:\Users\emm75\Documents\BMI_771\BMI771_Project\lg_tiles TCGA-BH-A201-01Z-00-DX1.6D6E3224-50A0-45A2-B231-EEF27CA7EFD2 94697 121856 DONE
# C:\Users\emm75\Documents\BMI_771\BMI771_Project\lg_tiles TCGA-D8-A1XY-01Z-00-DX1.AC051FB4-1D51-449B-BF2D-9DDB4382414C 46299 177072 DONE
# C:\Users\emm75\Documents\BMI_771\BMI771_Project\lg_tiles TCGA-D8-A1XZ-01Z-00-DX1.8E51A61D-B01C-4A52-8F5D-44D2ABCA46FC 66002 187544 DONE
# C:\Users\emm75\Documents\BMI_771\BMI771_Project\lg_tiles 
# C:\Users\emm75\Documents\BMI_771\BMI771_Project\lg_tiles 

# C:\Users\emm75\Documents\BMI_771\BMI771_Project\new_c8 TCGA-C8-A26V-01Z-00-DX1.6E86DF74-D575-4969-8142-963D9DF5208F 56013 75208

# DONE: for TCGA-A1-A0SE-01Z-00-DX1.04B09232-C6C4-46EF-AA2C-41D078D0A80A.svs : 120757 + 2082 / 72871 + 2082
# DONE: for TCGA-AR-A0TX-01Z-00-DX1.5BEA4E65-8CEC-49B2-9F29-DBE53E8ED46E.svs : x = 70192 + 2127 / y = 65938 + 2127
# DONE: for TCGA-A2-A0CV-01Z-00-DX1.B02D017A-61CD-45FE-BB31-705CD127DDAB.svs : 73062/159380
# DONE: for TCGA-B6-A0RP-01Z-00-DX1.02A55E9D-2AA9-497A-B481-85724EA813AD.svs : 87584/107533
# DONE: for TCGA-E2-A153-01Z-00-DX1.CA994467-E541-4131-A9FC-DCD9944F29C4.svs : 69021/80920
# DONE: for TCGA-B6-A0X5-01Z-00-DX1.02A9FF1E-EA20-4F2D-A2BB-427A287DE3FD.svs : 62423/123528
# DONE: for TCGA-BH-A0B6-01Z-00-DX1.4D982935-F600-4F37-ABB5-13AF74F4FDC4.svs : 77884/140765
# DONE: for TCGA-A2-A0T7-01Z-00-DX1.EA4DF9B5-8D04-4BCC-9ECB-CB8CB8ACBE1C.svs : 81052/99959

mask = sys.argv[1] + "\\" + sys.argv[2] + ".png"
img = Image.open(mask)

height = sys.argv[3]
width = sys.argv[4]

pixel = img.load()

folder_data = sys.argv[1] + "\\" + sys.argv[2] + ".svs"

print(mask,"\n",folder_data,"\n",height,width)

entries = os.scandir(folder_data)

count = 0

names = []

with os.scandir(folder_data) as entries:
    for entry in entries:
        coor = entry.name.split("_")
        #print(coor)
        s = coor[2].translate(' \n\t\r')
        dx = math.ceil(int(width)/int(s))
        dy = math.ceil(int(height)/int(s))
        cx = coor[0].translate(' \n\t\r')
        cy = coor[1].translate(' \n\t\r')
        new_cx = math.ceil(int(cx)/int(s))
        new_cy = math.ceil(int(cy)/int(s))
        #print(dx,dy,s,cx,cy,new_cx, new_cy)
        dif_x = img.size[0]/dx
        dif_y = img.size[1]/dy
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

#print(names)

with open((sys.argv[1] + "\\" + sys.argv[2] + ".svs/labels.txt"), 'w') as f:
    f.write('\n'.join(names))
