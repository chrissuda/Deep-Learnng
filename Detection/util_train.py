import wandb
import torch
from Loader import*
from util_detection import*
import copy

'''
This file provides several functions used in training the model.
'''

def evaluate(model,loader_val,device):
#return validation loss 	
	loss_val=0.0
	model=model.to(device=device)  # move the model parameters to CPU/GPU
	model.train()  # put model to training mode

	with torch.no_grad():
		for j,(x, y) in enumerate(loader_val):
			for i in range(len(y)):
				del y[i]['image_id']
				del y[i]["url"]
				#Put y into gpu/cpu
				y[i]={k:v.to(device) for k,v in y[i].items()}
			
			# move to device, e.g. GPU
			x=x.to(device=device, dtype=torch.float32)  
			score = model(x,y)
			
			loss_val+=sum(score.values())
			
	
	return loss_val/(j+1)


def train(model,optimizer,epochs,loader_train,loader_val,device,wb=False):
	if wb:
		wandb.init(name="original",project="detection")

	model=model.to(device=device)  # move the model parameters to CPU/GPU
	print("Device:",device)

	
	for e in range(1,epochs+1):
		loss_train=0

		start=time.time()
		#progress bar
		t=tqdm(total=len(loader_train))

		for j,(x,y) in enumerate(loader_train):
		
			for i in range(len(y)):
				del y[i]['image_id']
				del y[i]["url"]
				y[i]={k:v.to(device) for k,v in y[i].items()}

			model.train()  # put model to training mode
			x=x.to(device=device, dtype=torch.float32)  # move to device, e.g. GPU

			score = model(x,y)
			loss=sum(score.values())
			
			loss_train+=loss
			
			if wb:
				#log the loss
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
		t.close()
		end=time.time()
		
		loss_train/=(j+1)
		if(loader_val!=None):
			loss_val=evaluate(model,loader_val,device)
		else:
			loss_val=-1
		print("Epochs:",e," Time used:",int(end-start),"s",
		" loss_train:",loss_train.data," loss_val:",loss_val.data)
		#print("Scores_train:",{k:(v.data) for k,v in score.items()},'\n')

		#log the loss and loss_val
		if wb:
			wandb.log({"Epochs":e,"loss_train":loss_train.data,"loss_val":loss_val.data})
	
	if wb:
		torch.save(model, os.path.join(wandb.run.dir, "model.pt"))
	return model


def checkAp(model,loader_val,device,THRESHOLD_IOU=0.3,THRESHOLD_SCORE=0,THRESHOLD_NMS=0.4,THRESHOLD_SNMS=0.1,NMS=False):
	'''
	Check the Precision and Recall
	@param model a PyTorch model instance
	@param loader_val a Loader instance
	@param device torch.device(" ")
	@param THRESHOLD_IOU
	@param THRESHOLD_SCORE
	@param THRESHOLD_NMS
	@param THRESHOLD_SNMS used in snms() which only filter out same category
	@param NMS a boolean to indicate using Non-maximum suppression or not
	'''


	result={"door":{"index":[],"tp":0,"truth":0,"predict":0,"precision":0,"recall":0},
			"knob":{"index":[],"tp":0,"truth":0,"predict":0,"precision":0,"recall":0},
			"stair":{"index":[],"tp":0,"truth":0,"predict":0,"precision":0,"recall":0},
			"ramp":{"index":[],"tp":0,"truth":0,"predict":0,"precision":0,"recall":0}
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
					scoresPredict,labelsPredict,boxesPredict=nms(boxesPredict,labelsPredict,scoresPredict,THRESHOLD_NMS)
					scoresPredict,labelsPredict,boxesPredict=snms(boxesPredict,labelsPredict,scoresPredict,THRESHOLD_SNMS)

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
						result["door"]["predict"]+=1
						result["door"]["index"].append(n)
					elif labelsPredict[n]==2 and scoresPredict[n]>THRESHOLD_SCORE:
						result["knob"]["predict"]+=1
						result["knob"]["index"].append(n)
					elif labelsPredict[n]==3 and scoresPredict[n]>THRESHOLD_SCORE:
						result["stair"]["predict"]+=1
						result["stair"]["index"].append(n)
					elif labelsPredict[n]==4 and scoresPredict[n]>THRESHOLD_SCORE:
						result["ramp"]["predict"]+=1
						result["ramp"]["index"].append(n)
					

				#Update the number of ground_truth
				result["door"]["truth"]+=labelsTruth.count(1)
				result["knob"]["truth"]+=labelsTruth.count(2)
				result["stair"]["truth"]+=labelsTruth.count(3)
				result["ramp"]["truth"]+=labelsTruth.count(4)

				#Calculate tp(True Positive)
				for t,(k,v) in enumerate(result.items()):
					boxesA=[boxesTruth[n] for n in range(len(boxesTruth)) if (labelsTruth[n]==t+1)]
					boxesB= [boxesPredict[n] for n in range(len(boxesPredict)) if (n in v["index"])]
					v["tp"]+=checkTp(boxesA,boxesB,THRESHOLD_IOU)


	#calculate Precision and Recall
	print("*************************Recall Precision **************************************")
	for k,v in result.items():
		try:
			v["precision"]=float(v["tp"])/v["predict"]
			v["recall"]=float(v["tp"])/v["truth"]
			
		except ZeroDivisionError:
			print("The number of Predict or Truth is zero!")

		print(k.capitalize(),"-> TP:",v["tp"]," Predict:",v["predict"]," Truth:",v["truth"],
		" Precision:%.2f"%(100*v["precision"])+"%"," Recall:%.2f"%(100*v["recall"])+"%")

	print("*******************************************************************************\n")



def checkApOnBatch(target,y,device,THRESHOLD_IOU=0.3,THRESHOLD_SCORE=0,THRESHOLD_NMS=0.4,THRESHOLD_SNMS=0.1,NMS=False):
	'''
	Calculate the Precision and Recall given out ground_truth and predicted labels
	@param target predicted label
	@param y ground truth label
	@param device pytorch device using CPU or GPU
	@param THRESHOLD_IOU
	@param THRESHOLD_SCORE
	@param THRESHOLD_NMS 
	@param THRESHOLD_SNMS used in snms() which only filter out same category
	@param NMS a boolean variable to indicating using non-maximum suppression or not
	@return result a dict hold all the precision recall result ....
	'''


	result={"door":{"index":[],"tp":0,"truth":0,"predict":0,"precision":0,"recall":0},
			"knob":{"index":[],"tp":0,"truth":0,"predict":0,"precision":0,"recall":0},
			"stair":{"index":[],"tp":0,"truth":0,"predict":0,"precision":0,"recall":0},
			"ramp":{"index":[],"tp":0,"truth":0,"predict":0,"precision":0,"recall":0}
			}	

	
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
			scoresPredict,labelsPredict,boxesPredict=nms(boxesPredict,labelsPredict,scoresPredict,THRESHOLD_NMS)
			scoresPredict,labelsPredict,boxesPredict=snms(boxesPredict,labelsPredict,scoresPredict,THRESHOLD_SNMS)

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
				result["door"]["predict"]+=1
				result["door"]["index"].append(n)
			elif labelsPredict[n]==2 and scoresPredict[n]>THRESHOLD_SCORE:
				result["knob"]["predict"]+=1
				result["knob"]["index"].append(n)
			elif labelsPredict[n]==3 and scoresPredict[n]>THRESHOLD_SCORE:
				result["stair"]["predict"]+=1
				result["stair"]["index"].append(n)
			elif labelsPredict[n]==4 and scoresPredict[n]>THRESHOLD_SCORE:
				result["ramp"]["predict"]+=1
				result["ramp"]["index"].append(n)
			

		#Update the number of ground_truth
		result["door"]["truth"]+=labelsTruth.count(1)
		result["knob"]["truth"]+=labelsTruth.count(2)
		result["stair"]["truth"]+=labelsTruth.count(3)
		result["ramp"]["truth"]+=labelsTruth.count(4)

		#Calculate tp(True Positive)
		for t,(k,v) in enumerate(result.items()):
			boxesA=[boxesTruth[n] for n in range(len(boxesTruth)) if (labelsTruth[n]==t+1)]
			boxesB= [boxesPredict[n] for n in range(len(boxesPredict)) if (n in v["index"])]
			v["tp"]+=checkTp(boxesA,boxesB,THRESHOLD_IOU)


	#calculate Precision and Recall
	print("*************************Recall Precision **************************************")
	for k,v in result.items():
		try:
			v["precision"]=float(v["tp"])/v["predict"]
			v["recall"]=float(v["tp"])/v["truth"]
			
		except ZeroDivisionError:
			print("The number of Predict or Truth is zero!")

		print(k.capitalize(),"-> TP:",v["tp"]," Predict:",v["predict"]," Truth:",v["truth"],
		" Precision:%.2f"%(100*v["precision"])+"%"," Recall:%.2f"%(100*v["recall"])+"%")

	print("*******************************************************************************\n")

	return result