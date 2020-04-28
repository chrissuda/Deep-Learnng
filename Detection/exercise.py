import torch
import torchvision
from torchvision import transforms
from util_detection import*
from CocoFormat import*
import random
import time
import os

#load the model and predicts on an image
def result():

	model=torch.load("../model.pt") #model will be in cuda 
	model.eval()
	transform = transforms.Compose([transforms.ToTensor()])	

	path="../result"
	name=os.listdir(path)
	for n in name:
		img_path=os.path.join(path,n)
		img=Image.open(img_path).convert("RGB")
		x=transform(img)
		x=x.unsqueeze(0)
		x=x.cuda()
		target=model(x)
		
		#put x back to cpu
		x=x.cpu()
		draw(x[0],target[0],"Labelbox",file=os.path.join(path,"predict_"+n))

		


	# x=transform(img)
	# x=x.cuda()
	# print(x.size())
	# target=model(x)
	# draw(x,target,os.path.join(path,"predict_"+n))

result()

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

	transform = transforms.Compose([
					transforms.Resize(newSize),
					transforms.ToTensor()])

	labelbox=labelboxCoco(root,annFile,newSize,transform=transform)
	data=loader(labelbox,7,shuffle=True)

	#Set up the model
	model=torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
	#model=transfer(model,5)
	for param in model.parameters():
			param.requires_grad = True

	optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,betas=(0.9, 0.9))
	
	x1,y1=next(data)
	print("truth0:",y1[0],'\n')
	print("truth1:",y1[1],'\n')

	draw(x1[0],y1[0],"Labelbox",file="truth0.jpg")
	draw(x1[1],y1[1],"Labelbox",file="truth1.jpg")

	model.eval()
	target=model(x1)
	print('\n',"target0:",target[0])
	print('\n',"target1:",target[1])

	model.train()  # put model to training mode
	for i in range(3):
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

		print("loss:",loss.item())

	model.eval()
	target=model(x1)
	print('\n',"target0:",target[0])
	print('\n',"target1:",target[1])
	draw(x1[0],target[0],"Labelbox",file="predict0.jpg")
	draw(x1[1],target[1],"Labelbox",file="predict1.jpg")


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
	model=torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,)
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
