import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as im
import os
from Coco import Coco

def loader(data,batch_size):
    for i in range(0,len(data)-1,batch_size):
        img,label=[],[]
        for j in range(i,i+batch_size):
            x=data[j][0]
            img.append(x)
            
            y=data[j][1]
            label.append(y)
            
        img=torch.stack(img)
        yield img,label
        
annFile="/home/chris/cnn/coco/annotations/instances_val2017.json"
root="/home/chris/cnn/coco/val2017"
transform = T.Compose([
                    T.Resize((224,224)),
                    T.ToTensor()])


coco_val=Coco(root,annFile,transform=transform)

coco_loader=loader(coco_val,5)
x,y=next(coco_loader)
