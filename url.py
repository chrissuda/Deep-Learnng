import json
from urllib.request import urlretrieve
import os
import requests
folder="./img"


#Delete some unuseful information from labelbox data.
#Total:489 
def manipulate():

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
	for index in range(len(labelbox)):
		u=labelbox[index]['Labeled Data']
		i=labelbox[index]["External ID"]

		if labelbox[index]["Label"]=="Skip" or i in ID:
			delete.append(index)

		else:
			URL.append(labelbox[index]['Labeled Data'])
			ID.append(labelbox[index]["External ID"])
			label.append(labelbox[index])

	print("id:",len(ID))
	print("url:",len(URL))
	print("label",len(label))
	print("delete",len(delete))
	with open("label.json","w") as f:
		json.dump(label,f,indent=2)	

	

		# if(i in ID):
		# 	index=ID.index(i)
		# 	repeat.append({"ID":i,"Label_old":labelbox[index]["Label"],"Label_new":label["Label"]})
		# else:
		# 	URL.append(label['Labeled Data'])
		# 	ID.append(label["External ID"])
		# uSet.add(u)
		# iSet.add(i)
		# idSet.add(label["ID"])



def download():
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