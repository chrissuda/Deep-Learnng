'''
This file provides several functions used in training the model.
'''

import wandb
import torch
from Loader import*
from util_detection import*
from util_filter import *
import time
from tqdm import tqdm
import copy


def getValLoss(model,loader_val,device):
	"""calculate the validation  loss

	Args:
		model : Faster RCNN model
		loader_val : a generator which loads the validation data
		device : torch.device("cuda") or torch.device("cpu")

	Returns:
		torch float Tensor: the validation loss
	"""

	model=model.to(device=device)  # move the model parameters to CPU/GPU
	model.train()  # put model to training mode

	loss_val=0.0
	iterCount=0
	with torch.no_grad():
		for iterCount,(x, y) in enumerate(loader_val):
			for i in range(len(y)):
				#Put y into gpu/cpu
				y[i]={k:v.to(device) for k,v in y[i].items() if k in ["boxes","labels"]}
			
			# move to device, e.g. GPU
			x=x.to(device=device, dtype=torch.float32)

			#get the loss output  
			loss = model(x,y)
			
			#Add classification loss, objective loss and bounding box loss together
			loss_val+=sum(loss.values())
			
	
	return loss_val/(iterCount+1)



def train(model,optimizer,epochs,loader_train,loader_val,device,wb=False):
	"""train the model

	Args:
		model (): Faster RCNN model
		optimizer (): Pytorch optimizer
		epochs (): the number of iterations of looping training data
		loader_train (): a generator which loads the training data
		loader_val (): a generator which loads the validation data
		device (): torch.device("cuda") or torch.device("cpu")
		wb (bool, optional): whether to use wandb to log the result. Defaults to False.

	Returns:
		model: a model after being trained
	"""
	if wb:
		wandb.init(name="original",project="detection")

	model=model.to(device=device)  # move the model parameters to CPU/GPU
	print("Device:",device)

	
	for e in range(1,epochs+1):
		

		start=time.time()
		#progress bar
		t=tqdm(total=len(loader_train))

		loss_train=0
		iterCount=0
		for iterCount,(x,y) in enumerate(loader_train):
			for i in range(len(y)):
				#Put y into gpu/cpu
				y[i]={k:v.to(device) for k,v in y[i].items() if k in ["boxes","labels"]}

			model.train()  # put model to training mode
			x=x.to(device=device, dtype=torch.float32)  # move to device, e.g. GPU

			#output the training loss
			loss = model(x,y)
			loss=sum(loss.values())
			
			#Add training loss to the total training loss
			loss_train+=loss
			
			#log the loss
			if wb:
				wandb.log({"loss":loss})
			
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
			t.update(loader_train.batch_size)


		#Close progress bar
		t.close()
		end=time.time()
		
		#Calculate average training loss
		loss_train/=(iterCount+1)

		if(loader_val!=None):
			loss_val=getValLoss(model,loader_val,device)

			nonFilterAp=getAp(model,loader_val,device,NMS=True)
			printAp(nonFilterAp)


			filterAp=getAp(model,loader_val,device,NMS=True,isFilter=True)
			print("Filter:")
			printAp(filterAp)
		else:
			loss_val=-1

		print("Epochs:",e," Time used:",int(end-start),"s",
		" loss_train:",loss_train.data," loss_val:",loss_val.data,'\n')

		#log the loss and loss_val
		if wb:
			wandb.log({"Epochs":e,"loss_train":loss_train.data,"loss_val":loss_val.data})
	
	#Save the model to wandb cloud
	if wb:
		torch.save(model, os.path.join(wandb.run.dir, "model.pt"))

	return model


def printAp(result):
	"""print the precision and recall of each category

	Args:
		result (dict): a dict output by checkAp()
	"""

	print("*************************Recall Precision **************************************")
	for k,v in result.items():
		print(k.capitalize(),"-> TP:",v["tp"]," Predict:",v["predict"]," Truth:",v["truth"],
		" Precision:%.2f"%(100*v["precision"])+"%"," Recall:%.2f"%(100*v["recall"])+"%")

	print("*******************************************************************************")

def getAp(model,loader_val,device,THRESHOLD_IOU=0.3,THRESHOLD_SCORE=0,
		THRESHOLD_NMS=0.4,THRESHOLD_SNMS=0.1,NMS=False,isFilter=False):
	"""calculate the precision and recall of the current data

	Args:
		model : Faster RCNN model
		loader_val : a generator which loads the validation data
		device : torch.device("cuda") or torch.device("cpu")
		THRESHOLD_IOU (float, optional): . Defaults to 0.3.
		THRESHOLD_SCORE (int, optional): . Defaults to 0.
		THRESHOLD_NMS (float, optional): iou_threshold. Defaults to 0.4.
		THRESHOLD_SNMS (float, optional): iou_threshold. Defaults to 0.1.
		NMS (bool, optional): whether apply non-maximum suppression or not. Defaults to False.
		isFilter (bool, optional): whether apply filtering door algorithm using point cloud information. Defaults to False.

	Returns:
		dict: containg the recall and precision result corresponding 
				to each category
	"""


	result={"Door":{"index":[],"tp":0,"truth":0,"predict":0,"precision":0,"recall":0},
			"Knob":{"index":[],"tp":0,"truth":0,"predict":0,"precision":0,"recall":0},
			"Stairs":{"index":[],"tp":0,"truth":0,"predict":0,"precision":0,"recall":0},
			"Ramp":{"index":[],"tp":0,"truth":0,"predict":0,"precision":0,"recall":0}
			}	

	
	#Put model in evalation mode
	model.eval()
	model=model.to(device=device)  # move the model parameters to CPU/GPU

	for param in model.parameters():
		param.requires_grad = True
	
	
	with torch.no_grad():
		for i,(x, y) in enumerate(loader_val):
			# move to device, e.g. GPU
			x=x.to(device=device, dtype=torch.float32)  
			target = model(x)
			
			#Loop over an image batch
			for j in range(len(y)):
				#Set index to null;
				for k,v in result.items():
					v["index"]=[]

				#Extract boxes,labels and scores from Predict Labels
				labelsPredict=target[j]["labels"].tolist()
				boxesPredict=target[j]["boxes"].tolist()
				scoresPredict=target[j]["scores"].tolist()

				if NMS:
					#Apply NMS
					boxesPredict,labelsPredict,scoresPredict=nms(boxesPredict,labelsPredict,scoresPredict,THRESHOLD_NMS)
					boxesPredict,labelsPredict,scoresPredict=snms(boxesPredict,labelsPredict,scoresPredict,THRESHOLD_SNMS)
					
				if isFilter:
					
				# 	boxesPredict,labelsPredict,scoresPredict=removeDoors(boxesPredict,labelsPredict,scoresPredict,
				# image_id=y[j]["image_id"],folder="/home/students/cnn/NYC_PANO")
					boxesPredict,labelsPredict,scoresPredict=filterDoor(boxesPredict,labelsPredict,scoresPredict,
				(1000,1000),y[j]["image_id"],folder="/home/students/cnn/NYC_PANO")
					
				# 	boxesPredict,labelsPredict,scoresPredict=Filter(boxesPredict,labelsPredict,scoresPredict)
					
				

				#sort the confidence from highest to lowest
				sort_index=[s[0] for s in sorted(enumerate(scoresPredict), key=lambda x:x[1],reverse=True)]
				scoresPredict=[scoresPredict[s] for s in sort_index]
				labelsPredict=[labelsPredict[s] for s in sort_index]
				boxesPredict=[boxesPredict[s] for s in sort_index]

				#Extract boxes and labels from Ground Truth
				labelsTruth=y[j]["labels"].tolist()
				boxesTruth=y[j]["boxes"].tolist()

				#Loop over each predict labels in an image
				for n in range(len(labelsPredict)):

					if labelsPredict[n]==1 and scoresPredict[n]>THRESHOLD_SCORE:
						result["Door"]["predict"]+=1
						result["Door"]["index"].append(n)
					elif labelsPredict[n]==2 and scoresPredict[n]>THRESHOLD_SCORE:
						result["Knob"]["predict"]+=1
						result["Knob"]["index"].append(n)
					elif labelsPredict[n]==3 and scoresPredict[n]>THRESHOLD_SCORE:
						result["Stairs"]["predict"]+=1
						result["Stairs"]["index"].append(n)
					elif labelsPredict[n]==4 and scoresPredict[n]>THRESHOLD_SCORE:
						result["Ramp"]["predict"]+=1
						result["Ramp"]["index"].append(n)
					

				#Update the number of ground_truth
				result["Door"]["truth"]+=labelsTruth.count(1)
				result["Knob"]["truth"]+=labelsTruth.count(2)
				result["Stairs"]["truth"]+=labelsTruth.count(3)
				result["Ramp"]["truth"]+=labelsTruth.count(4)

				#Calculate tp(True Positive)
				for t,(k,v) in enumerate(result.items()):
					boxesA=[boxesTruth[n] for n in range(len(boxesTruth)) if (labelsTruth[n]==t+1)]
					boxesB= [boxesPredict[n] for n in range(len(boxesPredict)) if (n in v["index"])]
					v["tp"]+=getTp(boxesA,boxesB,THRESHOLD_IOU)


	#calculate recall and precision
	for k,v in result.items():
		try:
			v["precision"]=float(v["tp"])/v["predict"]
			v["recall"]=float(v["tp"])/v["truth"]
			
		except ZeroDivisionError:
			v["precision"]=0
			v["recall"]=0


	return result

# def checkApOnBatch(target,y,device,THRESHOLD_IOU=0.3,THRESHOLD_SCORE=0,THRESHOLD_NMS=0.4,THRESHOLD_SNMS=0.1,NMS=False,Filter=False):
# 	'''
# 	Calculate the Precision and Recall given out ground_truth and predicted labels
# 	@param target predicted label
# 	@param y ground truth label
# 	@param device pytorch device using CPU or GPU
# 	@param THRESHOLD_IOU
# 	@param THRESHOLD_SCORE
# 	@param THRESHOLD_NMS 
# 	@param THRESHOLD_SNMS used in snms() which only filter out same category
# 	@param NMS a boolean variable to indicating using non-maximum suppression or not
# 	@param Filter a boolean variable to indicate applying filtering or not
# 	@return result a dict hold all the precision recall result ....
# 	'''


# 	result={"door":{"index":[],"tp":0,"truth":0,"predict":0,"precision":0,"recall":0},
# 			"knob":{"index":[],"tp":0,"truth":0,"predict":0,"precision":0,"recall":0},
# 			"stair":{"index":[],"tp":0,"truth":0,"predict":0,"precision":0,"recall":0},
# 			"ramp":{"index":[],"tp":0,"truth":0,"predict":0,"precision":0,"recall":0}
# 			}	

	
# 	#Loop over an image batch
# 	for j in range(len(y)):
# 		#Set index to null;
# 		for k,v in result.items():
# 			v["index"]=[]

# 		#Extract boxes,labels and scores from Predict Labels
# 		labelsPredict=target[j]["labels"].tolist()
# 		boxesPredict=target[j]["boxes"].tolist()
# 		scoresPredict=target[j]["scores"].tolist()

# 		if NMS:
# 			#Apply NMS
# 			boxesPredict,labelsPredict,scoresPredict=nms(boxesPredict,labelsPredict,scoresPredict,THRESHOLD_NMS)
# 			boxesPredict,labelsPredict,scoresPredict=snms(boxesPredict,labelsPredict,scoresPredict,THRESHOLD_SNMS)

# 		if Filter:
# 					boxesPredict,labelsPredict,scoresPredict=Filter(boxesPredict,labelsPredict,scoresPredict)

# 		#sort the confidence from highest to lowest
# 		sort_index=[s[0] for s in sorted(enumerate(scoresPredict), key=lambda x:x[1],reverse=True)]
# 		scoresPredict=[scoresPredict[s] for s in sort_index]
# 		labelsPredict=[labelsPredict[s] for s in sort_index]
# 		boxesPredict=[boxesPredict[s] for s in sort_index]

# 		#Extract boxes and labels from Ground Truth
# 		labelsTruth=y[j]["labels"].tolist()
# 		boxesTruth=y[j]["boxes"].tolist()

# 		#Loop over each predict labels in an image
# 		for n in range(len(labelsPredict)):

# 			if labelsPredict[n]==1 and scoresPredict[n]>THRESHOLD_SCORE:
# 				result["door"]["predict"]+=1
# 				result["door"]["index"].append(n)
# 			elif labelsPredict[n]==2 and scoresPredict[n]>THRESHOLD_SCORE:
# 				result["knob"]["predict"]+=1
# 				result["knob"]["index"].append(n)
# 			elif labelsPredict[n]==3 and scoresPredict[n]>THRESHOLD_SCORE:
# 				result["stair"]["predict"]+=1
# 				result["stair"]["index"].append(n)
# 			elif labelsPredict[n]==4 and scoresPredict[n]>THRESHOLD_SCORE:
# 				result["ramp"]["predict"]+=1
# 				result["ramp"]["index"].append(n)
			

# 		#Update the number of ground_truth
# 		result["door"]["truth"]+=labelsTruth.count(1)
# 		result["knob"]["truth"]+=labelsTruth.count(2)
# 		result["stair"]["truth"]+=labelsTruth.count(3)
# 		result["ramp"]["truth"]+=labelsTruth.count(4)

# 		#Calculate tp(True Positive)
# 		for t,(k,v) in enumerate(result.items()):
# 			boxesA=[boxesTruth[n] for n in range(len(boxesTruth)) if (labelsTruth[n]==t+1)]
# 			boxesB= [boxesPredict[n] for n in range(len(boxesPredict)) if (n in v["index"])]
# 			v["tp"]+=getTp(boxesA,boxesB,THRESHOLD_IOU)


# 	#calculate Precision and Recall
# 	print("*************************Recall Precision **************************************")
# 	for k,v in result.items():
# 		try:
# 			v["precision"]=float(v["tp"])/v["predict"]
# 			v["recall"]=float(v["tp"])/v["truth"]
			
# 		except ZeroDivisionError:
# 			print("The number of Predict or Truth is zero!")

# 		print(k.capitalize(),"-> TP:",v["tp"]," Predict:",v["predict"]," Truth:",v["truth"],
# 		" Precision:%.2f"%(100*v["precision"])+"%"," Recall:%.2f"%(100*v["recall"])+"%")

# 	print("*******************************************************************************\n")

# 	return result
