import json
from urllib.request import urlretrieve
import os
import requests

#Delete some unuseful information from labelbox data.
#produce delLabelbox.json
def delete():
	folder="./img";

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
	print("after deleting, the number of total useful images is:",count);
	with open("labelbox.json","w") as f:
		json.dump(label,f,indent=2)	


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
        
# Change data into tensor format;
# boxes=torch.as_tensor(boxes,dtype=torch.float32)
# labels=torch.as_tensor(labels,dtype=torch.int64)
# iscrowd=torch.as_tensor(iscrowd,dtype=torch.uint8)

            target={"image_id":external_id,
            "boxes":boxes,"labels":labels,
            "iscrowd":iscrowd,"url":url}

        labelboxCoco.append(target)

    with open("labelboxCoco.json","w") as f:
        json.dump(labelboxCoco,f,indent=2)
    print(len(labelboxCoco))


#download images based on url
def download():
	folder="./img"
	i=0
	for label in labelbox:
		url=label['Labeled Data']
		path=folder+"/"+label["External ID"]
		urlretrieve(url,path)
		i+=1
	print("I:",i) 



def test():
	ID=[]
	URL=[]
	LABEL=[]
	repeat=[]
	uSet=set()
	iSet=set()
	idSet=set()

	for label in labelbox:
		u=label['Labeled Data']
		i=label["External ID"]

		if(i in ID):
			index=ID.index(i)
			#repeat.append({"ID":i,"URL_old":u,"URL_new":URL[index]})
			repeat.append({"ID":i,"Label_old":labelbox[index]["Label"],"Label_new":label["Label"]})
		else:
			URL.append(label['Labeled Data'])
			ID.append(label["External ID"])
		uSet.add(u)
		iSet.add(i)
		idSet.add(label["ID"])

	print("id:",len(ID))
	print("url:",len(URL))
	print("uSet:",len(uSet))
	print("iSet:",len(iSet))
	print("repeat:",len(repeat))
	print(len(idSet))
	for i in repeat[:10]:
		print(i,"\n")