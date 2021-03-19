import sys
sys.path.insert(0,"/home/students/cnn/Deep-Learnng")
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
from Detection.util_detection import*


def getMeanStd(loader):

    tensorsList=[]
    for i,(x,y) in enumerate(loader):
        tensorsList.append(x)

    tensors=torch.cat(tensorsList,dim=0)
    mean=torch.mean(tensors,dim=(0,2,3))
    std=torch.std(tensors,dim=(0,2,3))

    return mean,std

def turnIntoRelativeBoxes(file,image_size):
    new_size=(1,1)
    
    with open(file,'r') as f:
        data=json.load(f)
        
    for i in range(len(data)):
        boxes=data[i]["boxes"]
        boxes=toTensor(boxes)
        boxes=resizeBoxes(boxes,image_size,new_size)
        boxes=toList(boxes)
        data[i]["boxes"]=boxes

    with open(file,'w') as f:
        json.dump(data,f,indent=2)

# file="/home/students/cnn/Deep-Learnng/Data/UCFcoco.json"
# image_size=(1280,1024)
# turnIntoRelativeBoxes(file,image_size)
