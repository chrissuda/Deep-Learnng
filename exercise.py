import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as im
import os
from Coco import *
import json

def a():
	root="./labelbox_img"
	annFile="labelboxCoco.json"

	transform = T.Compose([T.Resize((800,800)),T.ToTensor()])
	        
	labelbox=labelboxCoco(root,annFile,(800,800),transform=transform)
	labelbox_loader=loader(labelbox,3)
	x,y=next(labelbox_loader)
	print(y[0])
	draw(x[0],y[0],"Labelbox","truth0.jpg")

def labelbox():
	root="./labelbox_img"
	annFile="labelboxCoco.json"

	transform = T.Compose([T.Resize((800,800)),T.ToTensor()])
	                

	labelbox=labelboxCoco(root,annFile,newSize=(800,800),transform=transform)

	model=torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

	labelbox_loader=loader(labelbox,3)
	x,y=next(labelbox_loader)

	model.eval()
	target=model(x)
	draw(x[0],target[0],"Coco","predict0.jpg")
	draw(x[0],y[0],"Labelbox","truth0.jpg")
	draw(x[1],target[1],"Coco","predict1.jpg")
	draw(x[1],y[1],"Labelbox","truth1.jpg")
	draw(x[2],target[2],"Coco","predict2.jpg")
	draw(x[2],y[2],"Labelbox","truth2.jpg")

labelbox()