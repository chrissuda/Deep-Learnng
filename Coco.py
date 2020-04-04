import torch
import matplotlib.pyplot as plt
import matplotlib.image as im
import os
from PIL import Image
from pycocotools.coco import COCO
import numpy as np


class Coco(torch.utils.data.Dataset):
    #transform: x  #target_transform: y  #transforms: (x,y)
    def __init__(self,img_root,annFile,
        transform=None,target_transform=None,transforms=None):

        super().__init__()
        self.coco=COCO(annFile)
        self.img_root=img_root
        self.imgs=list(sorted(os.listdir(img_root)))
        self.transform=transform
        self.target_transform=target_transform
        self.transforms=transforms


    def __getitem__(self,idx):
        img_name=self.imgs[idx]
        img_id=os.path.splitext(img_name)[0]
        img_id=int(img_id)
        annId=self.coco.getAnnIds(img_id)
        #annotation is a list containing various dict,
        #each dict is corresponding to a segementation in an image.
        annotation=self.coco.loadAnns(annId) #coresponding to an individual image

        #loop over an annotation
        boxes,labels,iscrowd=[],[],[]
        for a in annotation:
            labels.append(a["category_id"])
            iscrowd.append(a["iscrowd"])
            xmin=a["bbox"][0]
            xmax=a["bbox"][0]+a["bbox"][2]
            ymin=a["bbox"][1]
            ymax=a["bbox"][1]+a["bbox"][3]
            boxes.append([xmin,ymin,xmax,ymax])

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
            img=self.transform(img)
        if self.target_transform:
            print("target_transform")
            target=self.target_transform(target)
        if self.transforms:
            print("transforms")
            img,target=self.transforms(img,target)

      
        return img,target

    def __len__(self):
        return (len(self.imgs))







