import torch
import torchvision
import numpy as np
from torchvision import transforms
#from util_detection import*
# from util_train import*
# from CocoFormat import*
# import random
# import time
# import os
# from Loader import Loader

a=np.array([[1., -1.], [1., -1.]])
b=np.array([1,1])
print(a.dtype)

print(b.dtype)
x=torch.from_numpy(a)
y=torch.from_numpy(b)
print(x.type())
print(y.type())
# NUM_VAL=100
# model_path="../model.pt"
# batch_size=5
# annFile="labelboxCoco.json"
# root="../images"
# newSize=(800,800)
# transform = transforms.Compose([
#                 transforms.Resize(newSize),
#                 transforms.ToTensor()])
# labelbox=labelboxCoco(root,annFile,newSize,transform=transform)
# loader_val=Loader(labelbox,start=len(labelbox)-NUM_VAL,batch_size=batch_size,shuffle=False)
# checkAp(torch.load("../original.pt"),loader_val)
#predictInImageFolder("../NYC","../original.pt",IoUThreshold=0.2)

def label():
	
	root="../images"
	annFile="labelboxCoco.json"

	transform = T.Compose([T.Resize((800,800)),T.ToTensor()])
					
	labelbox=labelboxCoco(root,annFile,newSize=(800,800),transform=transform)
	l=labelbox[:300]
	model=torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
	
	#labelbox_loader=loader(labelbox,1)

	#x,y=next(labelbox_loader)

	model.eval()
	summary(model,input_size=(3,224,224))
	
def exercise():
	annFile="labelboxCoco.json"
	root="../images"
	newSize=(800,800)
	NUM_VAL=100
	batch_size=5

	transform = transforms.Compose([
					transforms.Resize(newSize),
					transforms.ToTensor()])

	labelbox=labelboxCoco(root,annFile,newSize,transform=transform)
	loader_val=Loader(labelbox,start=len(labelbox)-NUM_VAL,batch_size=batch_size,shuffle=False)
	loader_val=iter(loader_val)
	#Set up the model
	modelOrg=torch.load("../original.pt")
	model50=torch.load("../resnet50only.pt")

	modelOrg.eval();
	model50.eval();
	for i in range(8):
		x,y=next(loader_val);
		x=x.cuda();
		targetOrg=modelOrg(x);
		target50=model50(x);
		x=x.cpu()

		draw(x[2],targetOrg[2],"Labelbox",file="predict"+str(i)+"_original.jpg");
		draw(x[2],target50[2],"Labelbox",file="predict"+str(i)+"_50.jpg");
	


def coco_exercise():
	root="/home/students/cnn/Coco/val2017"
	annFile="/home/students/cnn/Coco/instances_val2017.json"
	
	newSize=(800,800)

	transform = transforms.Compose([
					transforms.Resize(newSize),
					transforms.ToTensor()])

	coco=Coco(root,annFile,newSize,transform=transform)
	data=loader(coco,5,shuffle=True)

	#Set up the model
	model=torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
	#model=transfer(model,5)
	for param in model.parameters():
			param.requires_grad = True

	optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,betas=(0.9, 0.9))
	#optimizer = torch.optim.SGD(model.parameters(), lr=0.005,
								#momentum=0.9, weight_decay=0.0005)
	
	x1,y1=next(data)
	print("truth0:",y1[0],'\n')
	print("truth1:",y1[1],'\n')

	draw(x1[0],y1[0],"Coco",file="truth0.jpg")
	draw(x1[1],y1[1],"Coco",file="truth1.jpg")



	model.train()  # put model to training mode


	for i in range(5):
		score = model(x1,y1)
		loss=sum(score.values())
		
		# Zero out all of the gradients for the variables which the optimizer
		# will update.
		optimizer.zero_grad()

		# This is the backwards pass: compute the gradient of the loss with
		# respect to each  parameter of the model.
		loss.backward()

		# Actually update the parameters of the model using the gradients
		# computed by the backwards pass.
		optimizer.step()

		#print("loss:",loss.item())
		print("\nloss:",loss)

	model.eval()
	target=model(x1)

	draw(x1[0],target[0],"Coco",file="predict0.jpg")
	draw(x1[1],target[1],"Coco",file="predict1.jpg")


