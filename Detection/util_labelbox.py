import json
from urllib.request import urlretrieve
import os
from tqdm import tqdm
import collections

def delete(annotation_path):
	'''
	Read json data download from LabelBox website
	Delete some unuseful information(Mostly for SKIP category)
	@param input_path the annotatino you want to load from
	@param output_path the path you want to save to
	@return the number of annotations
	'''
	with open(input_path) as f:
		labelbox=json.load(f)
	print(len(labelbox))

	ID=[]
	URL=[]
	LABEL=[]
	repeat=[]
	uSet=set()
	iSet=set()
	idSet=set()
	delete=[]
	label=[]
	count=0;
	null=0;
	for index in range(len(labelbox)):
		u=labelbox[index]['Labeled Data']
		i=labelbox[index]["External ID"]
		l=labelbox[index]["Label"]
		if l=="Skip" or i in ID or len(i)==0:
			delete.append(index)

		else:
			URL.append(labelbox[index]['Labeled Data'])
			ID.append(labelbox[index]["External ID"])
			label.append(labelbox[index])
			count+=1;

	print("id:",len(ID),"url:",len(URL)," null:",null," label:",len(label))
	print("deleted images:",len(delete))
	print("after deleting, the number of total useful annotation is:",count);

	assert(len(ID)==len(URL))
	assert(len(URL)==len(label))
	assert(len(label)==count)

	with open(outut_path,"w") as f:
		json.dump(label,f,indent=2)

	return count;


def download(img_folder,annotation_path): 
	'''
	download images based on url provided in the .json file
	Precondition the annotation file is in Coco format requested by Faster RCNN
	@param img_folder
	@param annotation_path
	@return the number of downloaded images
	'''
	with open(annotation_path,'r') as f:
		labelbox=json.load(f)

	folder=img_folder
	count=0
	for label in tqdm(labelbox):
		url=label['url']
		path=folder+"/"+label["image_id"]
		if not os.path.exists(path):
			urlretrieve(url,path)
			count+=1

	print("Total images:",count)
	return count 

def turnintoCoco_2(input_file,output_file):
	'''
	Turn NYC labelbox data into Coco format
	@Precondition your dataset is organized in a specific manner as followed:
	    {"ID":"ckfv0ep3d0006246a9ialuqc8","DataRow ID":"ckfuihtdc0pec0rdp68i98yja","Labeled Data":"https://storage.labelbox.com/cjvii2o9ehvtw0804li72x0u9%2F11212501-44cc-b7bd-a826-e9e6ad9eee48-TAhKmMfAex974GGcmXr_8g_0.jpg?Expires=1603246171847&KeyName=labelbox-assets-key-1&Signature=tqspRj1vX_6dj8x2_JFOOCuUhxU","Label":{"objects":[{"featureId":"ckfwhnd9o0frx0yaj2bd3dq64","schemaId":"ckfkh9byn005w0z5b40n8ag11","title":"Door","value":"door","color":"#1CE6FF","bbox":{"top":994,"left":2778,"height":672,"width":632},"instanceURI":"https://api.labelbox.com/masks/feature/ckfwhnd9o0frx0yaj2bd3dq64?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjazZ6ZTd1NG9wZ2FrMGMxNjlvYm5na21sIiwib3JnYW5pemF0aW9uSWQiOiJjanZpaTJvOWVodnR3MDgwNGxpNzJ4MHU5IiwiaWF0IjoxNjAyMDM2NTcxLCJleHAiOjE2MDQ2Mjg1NzF9.eT0Myg45gUEz7FtrTBTlG6FjDqvX56_43C1WlVBaI6M","classifications":[{"featureId":"ckfwhnet3084m0zbwgbe681k8","schemaId":"ckfv5ov620xna0yay2c381o8j","title":"Type","value":"type","answer":{"featureId":"ckfwhnetk084n0zbw7hap56hv","schemaId":"ckfv5ov6k0xne0yay27nf05di","title":"Double","value":"double"}}]},{"featureId":"ckfwhnimq085c0zbwfg654d0w","schemaId":"ckfkh9byn005y0z5b4qvfeutn","title":"Knob","value":"knob","color":"#FF34FF","bbox":{"top":1291,"left":3007,"height":151,"width":168},"instanceURI":"https://api.labelbox.com/masks/feature/ckfwhnimq085c0zbwfg654d0w?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjazZ6ZTd1NG9wZ2FrMGMxNjlvYm5na21sIiwib3JnYW5pemF0aW9uSWQiOiJjanZpaTJvOWVodnR3MDgwNGxpNzJ4MHU5IiwiaWF0IjoxNjAyMDM2NTcxLCJleHAiOjE2MDQ2Mjg1NzF9.eT0Myg45gUEz7FtrTBTlG6FjDqvX56_43C1WlVBaI6M"}],"classifications":[]},"Created By":"ndenizturhan@gmail.com","Project Name":"Panorama Street View","Created At":"2020-10-04T11:15:55.000Z","Updated At":"2020-10-05T12:08:24.126Z","Seconds to Label":23.583999999999996,"External ID":"TAhKmMfAex974GGcmXr_8g_0.jpg","Agreement":null,"Benchmark Agreement":-1,"Benchmark ID":null,"Dataset Name":"NYC Panorama","Reviews":[],"View Label":"https://editor.labelbox.com?project=ckf5itpph91io0726gceikrix&label=ckfv0ep3d0006246a9ialuqc8"},

	@param input_file directory path to your Labelboxraw dataset
	@param output_file directory path that you want to save to
	@return the number of data in this dataset
	'''

	with open (input_file,'r') as f:
		annotation=json.load(f)


	image_ids=[]
	data=[]

	for i in annotation:
	
		#Skip it if doesn't contain data
		if len(i["Label"])==0: #{}
			continue;

		#Skip it if there are same images
		image_id=i["External ID"]
		if image_id in image_ids:
			continue
		image_ids.append(image_id)
			
		url=i["Labeled Data"]

		boxes=[]
		labels=[]
		
		for label in i["Label"]["objects"]:
			
			category=label["title"]
			if category=="Door":
				labels.append(1)
			elif category=="Knob":
				labels.append(2)
			elif category=="Stairs":
				labels.append(3)
			elif category=="Ramp":
				labels.append(4)
			else:
				print("Non existed category:",category)
				continue

			#boxes[x1,y1,x2,y2]
			bbox=label["bbox"] #{'top': 1006, 'left': 566, 'height': 240, 'width': 364}
			y1=bbox["top"]
			x1=bbox["left"]
			y2=bbox["top"]-bbox["height"]
			x2=bbox["left"]-bbox["width"]
			boxes.append([x1,y1,x2,y2])

		iscrowd=[0]*len(labels)
		data.append({"image_id":image_id,"boxes":boxes,"labels":labels,"iscrowd":iscrowd,"url":url})
		


	with open(output_file,"w") as f:
		json.dump(data,f,indent=2)

	print("length of Dataset:",len(data))
	return len(data)

def turnintoCoco_1(input_file,output_file):
	'''
	Turn labelbox data into Coco format
	Precondition your dataset is organized in a specific manner
	
	@param input_file directory path to your Labelboxraw dataset
	@param output_file directory path that you want to save to
	@return the number of data in this dataset
	'''

	labelboxCoco=[]
	with open(input_file) as f:
		labelboxes=json.load(f)

	for labelbox in labelboxes:

		category={"Door":1,"Knob":2,"Stairs":3,"Ramp":4}
		url=labelbox["Labeled Data"]
		external_id=labelbox["External ID"]
		Label=labelbox['Label']
		boxes,labels,iscrowd=[],[],[]

		for k,v in Label.items():
			for box in v:
				xys=box["geometry"]
				xmax=max([x["x"] for x in xys])
				xmin=min([x["x"] for x in xys])
				ymax=max([y["y"] for y in xys])
				ymin=min([y["y"] for y in xys])
			boxes.append([xmin,ymin,xmax,ymax])
			labels.append(category[k])
			iscrowd.append(0)
		

			target={"image_id":external_id,
			"boxes":boxes,"labels":labels,
			"iscrowd":iscrowd,"url":url}

		labelboxCoco.append(target)

	with open(output_file,"w") as f:
		json.dump(labelboxCoco,f,indent=2)

	print("length of Dataset:",len(labelboxCoco))
	return len(labelboxCoco)


def verify(img_folder,annotation_path):
	'''
	verify if the number of images,annotations are equal
	Precondition your dataset is orgranized as Coco format required by Faster RCNN
	@param img_folder
	@param annotation_path
	'''

	with open(annotation_path) as f:
		annotations=json.load(f)

	images=[f for f in os.listdir(img_folder) if f.endswith('.tif')]
	
	assert(len(images)==len(annotations))
	print("######Verify######")
	print("images:",len(images)," annotations:",len(annotations))
	print("Data verify successed!")


def count(file):
	'''
	Count how many objects in each category in your your Labelbox dataset
	Precondition your dataset is orgranized as Coco format required by Faster RCNN
	@param file the path to your annotation .json file
	'''
	with open(file) as f:
		annotations=json.load(f)

	annotationList=[label["labels"] for label in annotations]
	annotationList=sum(annotationList,[])
	numDict=collections.Counter(annotationList)
	
	print("Door:",numDict[1]," Knob:",numDict[2]," Stairs:",numDict[3]," Ramp:",numDict[4])
	print("Total Objects:",sum(numDict.values())," Total Annotations:",len(annotations))


# def count(loader):
# 	'''
# 	count how many objects are in a Labelbox loader
# 	@param loader a Pytorch dataset loader
# 	'''
# 	Door,Knob,Stairs,Ramp=0,0,0,0
# 	truth_labels=[0,0,0,0,0]
# 	Img=len(loader)
# 	for x,y in loader:
# 		for i in range(len(y)):
# 			labels=y[i]['labels']
# 			for label in labels:
# 				if label==1:
# 					Door+=1
# 				elif label==2:
# 					Knob+=1
# 				elif label==3:
# 					Stairs+=1
# 				elif label==4:
# 					Ramp+=1
# 				else:
# 					print("Invalid label")

# 	print("############Ground Truth Labels#########")
# 	print("Total Images:",Img)
# 	print("Door:",Door," Knob:",Knob," Stairs:",Stairs," Ramp:",Ramp,"\n")
