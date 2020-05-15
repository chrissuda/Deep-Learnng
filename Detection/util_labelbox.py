import json
from urllib.request import urlretrieve
import os
from tqdm import tqdm


#Delete some unuseful information from labelbox data.
#produce delLabelbox.json
#return the number of annotations
def delete():

	with open("labelbox.json") as f:
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

	with open("labelbox.json","w") as f:
		json.dump(label,f,indent=2)

	return count;


#download images based on url
#return the number of useful images
def download(): 
	with open("labelbox.json",'r') as f:
		labelbox=json.load(f)

	folder="../images"
	count=0
	for label in tqdm(labelbox):
		url=label['Labeled Data']
		path=folder+"/"+label["External ID"]
		if not os.path.exists(path):
			urlretrieve(url,path)
		count+=1
	print("Total images:",count)
	return count 

#Turn labelbox data into Coco format
def turnintoCoco():
    labelboxCoco=[]
    with open("labelbox.json") as f:
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

    with open("labelboxCoco.json","w") as f:
        json.dump(labelboxCoco,f,indent=2)

    print("lenght of labelboxCoco:",len(labelboxCoco))
    return len(labelboxCoco)

#verify if the number of images, 
#annotations,labelboxCoco are equal
def verify():


	with open("labelboxCoco.json") as f:
		annotations=json.load(f)

	path="../images"
	images=os.listdir(path)

	assert(len(images)==len(annotations))
	print("######Verify######")
	print("images:",len(images)," annotations:",len(annotations))
	print("Data verify successed!")

#Count how many objects in each category in Labelbox
def count():
	with open("labelboxCoco.json") as f:
		annotations=json.load(f)
	Door,Knob,Stairs,Ramp,totalAnno,totalObj=0,0,0,0,0,0

	for annotation in annotations:
		labels=annotation['labels']
		for label in labels:
			if label==1:
				Door+=1
			elif label==2:
				Knob+=1
			elif label==3:
				Stairs+=1
			elif label==4:
				Ramp+=1
			else:
				print("Invalid label")
			
			totalObj+=1

		totalAnno+=1
	
	print("Door:",Door," Knob:",Knob," Stairs:",Stairs," Ramp:",Ramp)
	print("Total Objects:",totalObj," Total Annotations:",totalAnno)

#count how many objects are in a Labelbox loader
def count(loader):
	Door,Knob,Stairs,Ramp=0,0,0,0
	truth_labels=[0,0,0,0,0]
	Img=len(loader)
	for x,y in loader:
		for i in range(len(y)):
			labels=y[i]['labels']
			for label in labels:
				if label==1:
					Door+=1
				elif label==2:
					Knob+=1
				elif label==3:
					Stairs+=1
				elif label==4:
					Ramp+=1
				else:
					print("Invalid label")

	print("############Ground Truth Labels#########")
	print("Total Images:",Img)
	print("Door:",Door," Knob:",Knob," Stairs:",Stairs," Ramp:",Ramp,"\n")
