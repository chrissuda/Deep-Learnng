import torch
import os
from PIL import Image
from pycocotools.coco import COCO #for COCO
import numpy as np
import json
from operator import itemgetter
from util_detection import resizeBoxes

class Coco(torch.utils.data.Dataset):
    #transform: x  #target_transform: y  #transforms: (x,y)
    def __init__(self,img_root,annFile,newSize,
        transform=None,target_transform=None,transforms=None):

        super().__init__()
        self.coco=COCO(annFile)
        self.img_root=img_root
        self.imgs=list(sorted(os.listdir(img_root)))
        self.transform=transform
        self.target_transform=target_transform
        self.transforms=transforms
        self.newSize=newSize

    def __getitem__(self,idx):
        img_name=self.imgs[idx]
        img_id=os.path.splitext(img_name)[0]
        img_id=int(img_id)
        annId=self.coco.getAnnIds(img_id)
        #annotation is a list containing various dict,
        #each dict is corresponding to a segementation in an image.
        annotation=self.coco.loadAnns(annId) #coresponding to an individual image

        #loop over an annotation
        num=len(annotation)
        boxes,labels,iscrowd=[None]*num,[None]*num,[None]*num
        i=0
        for a in annotation:
            labels[i]=a["category_id"]
            iscrowd[i]=a["iscrowd"]
            xmin=a["bbox"][0]
            xmax=a["bbox"][0]+a["bbox"][2]
            ymin=a["bbox"][1]
            ymax=a["bbox"][1]+a["bbox"][3]
            boxes[i]=[xmin,ymin,xmax,ymax]
            i+=1


    # Change data into tensor format;
        boxes=torch.as_tensor(boxes,dtype=torch.float32)
        labels=torch.as_tensor(labels,dtype=torch.int64)
        iscrowd=torch.as_tensor(iscrowd,dtype=torch.uint8)
        image_id=torch.full((len(annotation),),img_id,dtype=torch.int32)

        target={"boxes":boxes,"labels":labels,
                "iscrowd":iscrowd,"image_id:":image_id}
        
        img_path=os.path.join(self.img_root,img_name)
        img=Image.open(img_path).convert("RGB")
        
        if self.transform:
            originSize=img.size
            img=self.transform(img)
            target["boxes"]=resizeBoxes(boxes,originSize,self.newSize)
            
        if self.target_transform:
            print("target_transform")
            target=self.target_transform(target)
        if self.transforms:
            print("transforms")
            img,target=self.transforms(img,target)

      
        return img,target

    def __len__(self):
        return (len(self.imgs))

class labelboxCoco(torch.utils.data.Dataset):
    #transform: x  #target_transform: y  #transforms: (x,y)
    def __init__(self,img_root,annFile,newSize,
        transform=None,target_transform=None,transforms=None):

        super().__init__()

        labelbox=json.load(open(annFile))
        self.labelbox=sorted(labelbox, key=itemgetter('image_id'))
        self.image_root=img_root
        self.transform=transform
        self.target_transform=target_transform
        self.transforms=transforms
        self.newSize=newSize  #Shape(x,y) after reshape

    def __getitem__(self,idx):            
        label=self.labelbox[idx]
        image_id=label["image_id"]

    # Change data into tensor format;
        boxes=torch.as_tensor(label["boxes"],dtype=torch.float32)
        labels=torch.as_tensor(label["labels"],dtype=torch.int64)
        iscrowd=torch.as_tensor(label["iscrowd"],dtype=torch.uint8)

        target={"image_id":image_id,"boxes":boxes,"labels":labels,"iscrowd":iscrowd,
        "url":label["url"]}
        
        image_path=os.path.join(self.image_root,image_id)
        img=Image.open(image_path).convert("RGB")
        
        if self.transform:
            originSize=img.size
            img=self.transform(img)
            target["boxes"]=resizeBoxes(boxes,originSize,self.newSize)

      
        return img,target

    def __len__(self):
        return (len(self.labelbox))
    
    







