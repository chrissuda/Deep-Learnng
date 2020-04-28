import json
from urllib.request import urlretrieve
import os

#Delete some unuseful information from labelbox data.
#produce delLabelbox.json
#return the number of annotations
def delete():
	folder="../images";

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
	for index in range(len(labelbox)):
		u=labelbox[index]['Labeled Data']
		i=labelbox[index]["External ID"]
		l=labelbox[index]["Label"]
		if l=="Skip" or i in ID or len(l)==0 :
			delete.append(index)

		else:
			URL.append(labelbox[index]['Labeled Data'])
			ID.append(labelbox[index]["External ID"])
			label.append(labelbox[index])
			count+=1;

	print("id:",len(ID),"url:",len(URL),"label:",len(label))
	print("deleted images:",len(delete))
	print("after deleting, the number of total useful annotation is:",count);

	assert(len(ID)==len(URL))
	assert(len(URL)==len(label))
	assert(len(label)==len(count))

	with open("labelbox.json","w") as f:
		json.dump(label,f,indent=2)

	return count;


#download images based on url
#return the number of useful images
def download(): 
	folder="../images"
	count=0
	for label in labelbox:
		url=label['Labeled Data']
		path=folder+"/"+label["External ID"]
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
def verify(images,annotations,labelboxCoco):
	assert(images==annotations)
	assert(annotations==labelboxCoco)

	with open("labelboxCoco.json") as f:
		annotations=json.load(f)

	path="../images"
	images=len(os.listdir(path))

	print("######Verify######")
	print("images:",images," annotations:",annotations)
	print("Data verify successed!")
