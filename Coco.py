import torch
import matplotlib.pyplot as plt
import matplotlib.image as im
import os
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import json
from operator import itemgetter


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
            # labels.append(a["category_id"])
            # iscrowd.append(a["iscrowd"])
            # xmin=a["bbox"][0]
            # xmax=a["bbox"][0]+a["bbox"][2]
            # ymin=a["bbox"][1]
            # ymax=a["bbox"][1]+a["bbox"][3]
            # boxes.append([xmin,ymin,xmax,ymax])

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

#import json
#from operator import itemgetter
class labelboxCoco(torch.utils.data.Dataset):
    #transform: x  #target_transform: y  #transforms: (x,y)
    def __init__(self,img_root,annFile,
        transform=None,target_transform=None,transforms=None):

        super().__init__()

        labelbox=json.load(open(annFile))
        self.labelbox=sorted(labelbox, key=itemgetter('image_id'))
        self.image_root=img_root
        self.transform=transform
        self.target_transform=target_transform
        self.transforms=transforms


    def __getitem__(self,idx):            
        label=self.labelbox[idx]
        print(label)
        image_id=label["image_id"]

    # Change data into tensor format;
        boxes=torch.as_tensor(label["boxes"],dtype=torch.float32)
        labels=torch.as_tensor(label["labels"],dtype=torch.int64)
        iscrowd=torch.as_tensor(label["iscrowd"],dtype=torch.uint8)

        target={"image_id:":image_id,"boxes":boxes,"labels":labels,
                "iscrowd":iscrowd,"url":label["url"]}
        
        image_path=os.path.join(self.image_root,image_id)
        img=Image.open(image_path).convert("RGB")
        
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
        return (len(self.labelbox))


def loader(data,batch_size):
    for i in range(0,len(data)-1,batch_size):
        img,label=[None]*batch_size,[None]*batch_size
        for j in range(i,i+batch_size):
            img[j-i]=data[j][0]
            label[j-i]=data[j][1]
            
        img=torch.stack(img)
        yield img,label

#img: a tensor[c,h,w]
#target: a dict contains various boxes,labels   
def draw(img,target,file_name=None):

    #Open the categories file
    with open("/home/chris/cnn/Deep-Learnng/categories.json") as f:
        #It is a list contains dicts
        categories=json.load(f)['categories']

    #unpack target dict {"boxes":boxes,"labels":labels,......}
    boxes=target["boxes"].tolist() #convert tensor to list
    labels=target["labels"].tolist()
    labels=[categories[i]["name"] for i in labels]
    try: 
        scores=target["scores"].tolist()
        print("Image visualization based on model's predictation")
    except:
        print("Image visualization based on ground truth")

    #Convert tensor[c,h,w] to PIL image
    transform =torchvision.transforms.ToPILImage(mode='RGB')
    img=transform(img)

    
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf",10)   #, encoding="unic")
    
    #plt.ion() //Interactive mode
    # plt.ioff()
    # plt.imshow(img)
    # plt.axis('on')
    #plt.pause(0.2)    
    plt.figure(figsize=(30,20))
    draw=ImageDraw.Draw(img)
    for i in range(len(boxes)):
        r=random.randint(0,25)*10
        g=random.randint(0,25)*10
        b=random.randint(0,25)*10
        color=(r,g,b)
        draw.rectangle(boxes[i],outline=color,width=4) #[x0,y0,x1,y1]
        text=str(labels[i])
        
        draw.text((boxes[i][0],boxes[i][1]-11),text=text,fill=color,font=font)
    
    plt.imshow(img)
    plt.axis('on')
    if file_name!=None:
        plt.savefig(file_name)
    plt.show()
    print("pass")

def IoU(box1,box2):

    if box2[0]>box1[2] or box1[0]>box2[2] or box2[1]>box1[3] or box1[1]>box2[3]:
        return 0
    else:
        xmax=max(box1[0],box2[0])
        xmin=min(box1[2],box2[2])
        ymax=max(box1[1],box2[1])
        ymin=min(box1[3],box2[3])

        intersection=(xmin-xmax)*(ymin-ymax)
        area1=(box1[2]-box1[0])*(box1[3]-box1[1])
        area2=(box2[2]-box2[0])*(box2[3]-box2[1])

        iou=intersection/float(area1+area2-intersection)
        return iou


#boxes1:ground truth 
#boxes2:predictation
def allIoU(boxes1,boxes2):
    iou=[None]*len(boxes1)
    index=[None]*len(boxes1) #The index for which element is selected from boxes2
    for i in range(len(boxes1)):
        arr=[loU(boxes[i],boxes2) for j in range(len(boxes2))]
        iou[i]=max(arr)
        index[i]=arr.index(lou[i])
    return iou,index



#Delete some unuseful information from labelbox data.
#Total:489 
def delLabelbox():

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


#Turn labelbox data into Coco format
def turnintoCoco():
    labelboxCoco=[]
    with open("delLabelbox.json") as f:
        labelboxes=json.load(f)


    for labelbox in labelboxes:
        category={"Door":1,"Stairs":2,"Knob":3,"Ramp":4}
        url=labelbox["Labeled Data"]
        external_id=labelbox["External ID"]
        Label=labelbox['Label']
        boxes,labels,iscrowd=[],[],[]

        for k,v in Label.items():
            for box in v:
                xys=box["geometry"]
                xmax=max([x["x"] for x in xys])
                xmin=max([x["x"] for x in xys])
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
