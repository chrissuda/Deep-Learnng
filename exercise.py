#import torch
#import torch.nn as nn
#import torch.optim as optim
#import torchvision
#import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as im
import os
#from CocoFormat import *
import json
#from torchsummary import summary
import random
import time

def label():
	
	root="./labelbox_img"
	annFile="labelboxCoco.json"

	transform = T.Compose([T.Resize((800,800)),T.ToTensor()])
	                
	labelbox=labelboxCoco(root,annFile,newSize=(800,800),transform=transform)

	model=torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

	#labelbox_loader=loader(labelbox,1)

	#x,y=next(labelbox_loader)

	model.eval()
	summary(model,input_size=(3,224,224))
	
def exercise():
	a=time.time()
	time.sleep(1)
	end=time.time()
	print(int(end-a),"s")
exercise()




