from itertools import chain
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import json
import tqdm
from tqdm import tqdm
from PIL import Image,ImageFont,ImageDraw
import matplotlib.pyplot as plt
import random
import time
from util_labelbox import count
import os
import wandb

#num_classes: background+classes
def transfer(model,num_classes):
	#backbone = torchvision.models.resnet50(pretrained=True)
	#model.backbone=torch.nn.Sequential(*list(backbone.children())[:-2])
	#model.backbone.add_module("conv",torch.nn.Conv2d(2048,256,(1,1),(1,1)))
	#model.backbone.add_module("relu",torch.nn.ReLU())
	

	# get number of input features for the classifier
	in_features = model.roi_heads.box_predictor.cls_score.in_features
	# replace the pre-trained head with a new one
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
	
	return model



#Using cuda
def test(loader):
	#Calculate the truth-label first
	count(loader)

	doors,knobs,stairs,ramps=0,0,0,0 #predict labels
	#model=torch.load('../model.pt', map_location=lambda storage, loc: storage)
	model=torch.load("../model.pt")
	model.eval()
	for x,y_truth in loader:
		x=x.cuda()
		y_predict=model(x)
		for i in range(len(y_predict)):
			labels=y_predict[i]["labels"]
			boxes=y_predict[i]["boxes"]
			for label in labels:
				if label==1:
					doors+=1
				elif label==2:
					knobs+=1
				elif label==3:
					stairs+=1
				elif label==4:
					ramps+=1
				else:
					print("Invalid label")

	print("######Predict Labels:########")
	print("doors:",doors," knobs:",knobs," stairs:",stairs," ramps:",ramps)


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


# def loader(data,batch_size,shuffle=False,initital_index=0):
# 	List=list(range(initital_index,len(data)-1-batch_size,batch_size))
# 	if(shuffle):
# 		random.shuffle(List)

# 	for i in List:
# 		img=[data[j][0] for j in range(i,i+batch_size)]    
# 		img=torch.stack(img)
# 		label=[data[j][1] for j in range(i,i+batch_size)]


# 		yield img,label



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
	isTensor=torch.is_tensor(target["labels"]) #Verify if target[] is a tensor type or list type

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
	boxes=target["boxes"]
	labels=target["labels"]
	if isTensor:
		boxes=boxes.tolist() #convert tensor to list
		labels=labels.tolist()

	labels=[categories[i]["name"] for i in labels]

	try: 
		scores=target["scores"]
		if isTensor:
			scores=scores.tolist()
		scores=[":"+str(int(s*100))+"%" for s in scores]
		print("Image visualization based on model's predictation")
	except:
		scores=[""]*len(labels)
		print("Image visualization based on ground truth")


	#Convert tensor[c,h,w] to PIL image
	transform =torchvision.transforms.ToPILImage(mode='RGB')
	img=transform(img)
	font_size=int(img.size[0]*12.0/800)
	try:
		#Linux
		font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSerif.ttf",font_size)
	except:
		#Windows
	   font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf",font_size)
	
	draw=ImageDraw.Draw(img)

	for i in range(len(boxes)):
		try:
			if target["scores"][i]<0.1:
				continue;
		except:
			pass;
		
		#[x0,y0,x1,y1]
		draw.rectangle(boxes[i],outline=color[str(labels[i])],width=2) 
		text=str(labels[i])+scores[i]

		draw.text((boxes[i][0],boxes[i][1]+1),
			text=text,fill=color[str(labels[i])],font=font) 
	

	if file!=None:
		img.save(file)
		print("image has been saved")

	plt.imshow(img)
	plt.axis('on')
	plt.show()


def IoU(boxA,boxB):
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	interArea = max(0, xB - xA) * max(0, yB - yA)
	
	boxAArea = (boxA[2] - boxA[0] ) * (boxA[3] - boxA[1] )
	boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
	
	iou = interArea / float(boxAArea + boxBArea - interArea)

	return iou

#boxes1:ground truth 
#boxes2:predictation
def checkTp(boxes1,boxes2,threshold):
	#Store the index of a groudn truth if it's already corresponding to a predict label
	index=[]

	tp=0
	for box2 in boxes2: 
		for i in range(len(boxes1)):
			if IoU(boxes1[i],box2)>threshold and i not in index:
				index.append(i)
				tp+=1

	return tp

'''
load the model in "/Deep-Learnng.model.pt" 
and predicts using all the images from Deep-Learnng/result folder
dataset:"Coco" or "Labelbox"
'''
def predictInImageFolder(imges_path,model_path,IoUThreshold,dataset="Labelbox",NMS=True):
	
	model=torch.load(model_path) #model will be in cuda 

	model.eval()
	transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])	

	
	name=os.listdir(imges_path)
	for n in name:
		if not n.startswith("predict"):
			img_path=os.path.join(imges_path,n)
			img=Image.open(img_path).convert("RGB")
			x=transform(img)
			x=x.unsqueeze(0)
			x=x.cuda()
			target=model(x)

			#Using NMS
			if NMS:
				score=target[0]["scores"].tolist()
				label=target[0]["labels"].tolist()
				box=target[0]["boxes"].tolist()
				target[0]["scores"],target[0]["labels"],target[0]["boxes"]=nms(box,label,label,IoUThreshold)

			#put x back to cpu
			x=x.cpu()
			draw(x[0],target[0],"Labelbox",file=os.path.join(imges_path,"predict_"+n))




"""
    Non-max Suppression Algorithm 
    @param list  Object candidate bounding boxes 
    @param list  Confidence score of bounding boxes
    @param float IoU threshold
    @return list picked_score,picked_label,picked_boxes

"""
def nms(bounding_boxes,label,confidence_score,threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], [],[]

    # Bounding boxes
    boxes = toNumpy(bounding_boxes)
    # Confidence scores of bounding boxes
    score = toNumpy(confidence_score)


    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []
    picked_label= []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])
        picked_label.append(label[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_score,picked_label,picked_boxes


def toNumpy(x):

	typeClass=str(type(x))
	#If x is a Torch.tensor
	if "torch" in typeClass:
		x=x.detach().cpu().numpy()

	#If x is a list
	elif "list" in typeClass:
		x=np.asarray(x)
		

	return x

def toList(x):
	typeClass=str(type(x))

	#If x is a Torch.tensor
	if "torch" in typeClass:
		x=x.tolist()

	#If x is a numpy array
	elif "numpy" in typeClass:
		x=x.tolist()

	return x


def toTensor(x):
	typeClass=str(type(x))

	#If x is a list
	if "list" in typeClass:
		x=torch.Tensor(x)

	#If x is a numpy array
	elif "numpy" in typeClass:
		x=torch.from_numpy(x)

		#Manually change to float32 if it is float64
		if "float" in str(x.dtype):
			x=x.type(torch.float32)

	return x