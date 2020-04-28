from itertools import chain
import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import json
import tqdm
from tqdm import tqdm
from PIL import Image,ImageFont,ImageDraw
import matplotlib.pyplot as plt
import random
import time
import wandb
import os

#num_classes: background+classes
def transfer(model,num_classes):
	# replace the classifier with a new one, that has
	# num_classes which is user-defined

	# get number of input features for the classifier
	in_features = model.roi_heads.box_predictor.cls_score.in_features
	# replace the pre-trained head with a new one
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

	return model


def train(model,optimizer,data,batch_size,epochs=2):
	wandb.init(name="detection",project="pytorch")

	# Using GPU or CPU
	if torch.cuda.is_available():
		device = torch.device('cuda')
		torch.backends.cuda.cufft_plan_cache.clear()
	else:
		device = torch.device('cpu')
	model = model.to(device=device)  # move the model parameters to CPU/GPU
	print("Device:",device)

	
	for e in range(1,epochs+1):
		load=loader(data,batch_size,shuffle=True)
		start=time.time()
		#progress bar
		t=tqdm(total=int(len(data)/batch_size))

		for i,(x, y) in enumerate(load):
			
			for i in range(len(y)):
				del y[i]['image_id']
				del y[i]["url"]
				y[i]={k:v.to(device) for k,v in y[i].items()}

			model.train()  # put model to training mode
			x = x.to(device=device, dtype=torch.float32)  # move to device, e.g. GPU

			score = model(x,y)

			loss=sum(score.values())
			#wandb.log({"train_loss": loss}) #log the loss for each iteration
			
			# Zero out all of the gradients for the variables which the optimizer
			# will update.
			optimizer.zero_grad()

			# This is the backwards pass: compute the gradient of the loss with
			# respect to each  parameter of the model.
			loss.backward()

			# Actually update the parameters of the model using the gradients
			# computed by the backwards pass.
			optimizer.step()

			#update tqdm
			t.update(1)

			wandb.log({"loss":loss})

		t.close()
		end=time.time()
		print("Epochs:",e," Time used:",int(end-start),"s"," loss:",loss.item())
		print("scores:",score,'\n')

	
	return model


# def test():
#     root="./labelbox_img"
#     annFile="labelboxCoco.json"

#     transform = T.Compose([
#                     T.Resize((800,800)),
#                     T.ToTensor()])

#     labelbox=labelboxCoco(root,annFile,transform=transform)

#     model=torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

#     iou,confidence=[],[]
#     doorNum=0;
#     total=0
#     labelbox_loader=loader(labelbox,3)
#     model.eval()
#     #bar=tqdm(total=489)
#     for x,y in labelbox_loader:
#         predict=model(x)
		
#         for j in range(len(predict)): #Coresponding to an image
#             #Turn torch tensor into list
#             y[j]["boxes"]=y[j]["boxes"].tolist()
#             y[j]["labels"]=y[j]["labels"].tolist()
#             predict[j]["boxes"]=predict[j]["boxes"].tolist()
#             predict[j]["labels"]=predict[j]["labels"].tolist()
#             predict[j]["scores"]=predict[j]["scores"].tolist()

#             label=predict[j]["labels"]
#             index_predict=[i for i in range(len(label)) if label[i] == 71]

#             index_truth=[i for i in range(len(y[j]["labels"])) if y[j]["labels"][i] == 71]
#             doorNum+=len(index_predict)
#             print(index_truth)
#             if(len(index_predict)!=0 and len(index_truth!=0)):
#                 box=predict[j]["boxes"]
#                 box_predict=[box[i] for i in index_predict]
#                 box_truth=[y[j]["boxes"][i] for i in index_truth]
#                 box_index=set() #get the index where there is Door based on truth and prediction
#                 for b in range(len(box_truth)):
#                     box_result=[]
#                     for bb in range(len(box_predict)):
#                         box_result.append(IoU(box_truth[b],box_predict[bb]))    
#                     box_max=max(box_result)
#                     box_index.add(box_result.index(box_max))
#                     iou.append(box_max)

#                 score=predict[j]["scores"]
#                 score=[score[i] for i in box_index]
#                 confidence.append(score)

#         total+=1;
#         bar.update(3)

#     iouList=list(chain(*iou))
#     print("average_iou:",sum(iouList)/len(iouList))
#     confidenceList=list(chain(*iou))
#     print("average_confidence:",sum(confidenceList)/len(confidenceList))
#     #print("Total Door:",489," Predction:",doorNum)
#     #print("Prediction_accuracy:%.2f%",float(doorNum)*100.0/489)
#     print("Total iteration:",total)
#     dict={"iouList":iouList,"confidenceList":confidenceList}
#     with open("experiment.json","w") as f:
#         json.dump(dict,f)


def loader(data,batch_size,shuffle=False,initital_index=0):
	List=list(range(initital_index,len(data)-1-batch_size,batch_size))
	if(shuffle):
		random.shuffle(List)

	for i in List:
		img=[data[j][0] for j in range(i,i+batch_size)]    
		img=torch.stack(img)
		label=[data[j][1] for j in range(i,i+batch_size)]


		yield img,label



#originSize(x,y)
#newSize(x,y)
def resizeBoxes(boxes,originSize,newSize):
	ratioX=newSize[0]/originSize[0]
	boxes[:,0::2]*=ratioX

	ratioY=newSize[1]/originSize[1]
	boxes[:,1::2]*=ratioY
	return boxes;


#img: a tensor[c,h,w]
#target: a dict contains various boxes,labels
#dataset:"Coco" or "Labelbox"
def draw(img,target,dataset,file=None):

	#Open the categories file
	with open("categories.json") as f:
		#It is a list contains dicts
		categories=json.load(f)[dataset]

	if dataset=="Coco":
		print("Showing images on Coco dataset")
		color={k["name"]:(random.randint(0,25)*10,random.randint(0,25)*10,
		random.randint(0,25)*10) for k in categories}
	
	elif dataset=="Labelbox":
		print("Showing images on Labelbox dataset")
		color={"None":(0,0,0),"Door":(255,0,0), 
		"Knob":(0,200,255),"Stairs":(0,255,0),"Ramp":(255,182,193)}
		
	else:
		print("Invalid dataset\n")
		raise NameError('The dataset name is wrong in draw()')
	
	
	#unpack target dict {"boxes":boxes,"labels":labels,......}
	boxes=target["boxes"].tolist() #convert tensor to list
	labels=target["labels"].tolist()
	labels=[categories[i]["name"] for i in labels]
	try: 
		scores=target["scores"].tolist()
		scores=[":"+str(int(s*100))+"%" for s in scores]
		print("Image visualization based on model's predictation")
	except:
		scores=[""]*len(labels)
		print("Image visualization based on ground truth")


	#Convert tensor[c,h,w] to PIL image
	transform =torchvision.transforms.ToPILImage(mode='RGB')
	img=transform(img)
	font_size=int(img.size[0]*16.0/800)
	try:
		#Linux
		font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSerif.ttf",font_size)
	except:
		#Windows
	   font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf",font_size)
	
	draw=ImageDraw.Draw(img)

	for i in range(len(boxes)):
		#[x0,y0,x1,y1]
		draw.rectangle(boxes[i],outline=color[str(labels[i])],width=2) 
		text=str(labels[i])+scores[i]

		draw.text((boxes[i][0],boxes[i][1]+font_size-2),
			text=text,fill=color[str(labels[i])],font=font) 
	

	if file!=None:
		img.save(file)
		print("image has been saved")

	plt.imshow(img)
	plt.axis('on')
	plt.show()


def IoU(box1,box2):

	if box2[0]>box1[2] or box1[0]>box2[2] or box2[1]>box1[3] or box1[1]>box2[3]:
		return 0
	else:
		xmax=max(box1[0],box2[0])
		xmin=min(box1[2],box2[2])
		ymax=max(box1[1],box2[1])
		ymin=min(box1[3],box2[3])

		intersection=(xmin-xmax)*(ymin-ymax)
		area1=(box1[2]-box1[0])*(box1[3]-box1[1])
		area2=(box2[2]-box2[0])*(box2[3]-box2[1])

		iou=intersection/float(area1+area2-intersection)
		return iou


#boxes1:ground truth 
#boxes2:predictation
def allIoU(boxes1,boxes2):
	iou=[None]*len(boxes1)
	index=[None]*len(boxes1) #The index for which element is selected from boxes2
	for i in range(len(boxes1)):
		arr=[loU(boxes[i],boxes2) for j in range(len(boxes2))]
		iou[i]=max(arr)
		index[i]=arr.index(lou[i])
	return iou,index