from itertools import chain
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import json
import tqdm
from tqdm import tqdm
from PIL import Image,ImageFont,ImageDraw
import matplotlib.pyplot as plt
import random
import time
from util_labelbox import count
import os
import sys

def transfer(model,num_classes):
    """
    modify the structure of the model so than it can fit into your detection task

    Args:
        model (pytorch Faster RCNN ): 
        num_classes (int): the number of category will be detected in detection task

    Returns:
        model: a Faster RCNN model after modification
    """

    #backbone = torchvision.models.resnet50(pretrained=True)
    #model.backbone=torch.nn.Sequential(*list(backbone.children())[:-2])
    #model.backbone.add_module("conv",torch.nn.Conv2d(2048,256,(1,1),(1,1)))
    #model.backbone.add_module("relu",torch.nn.ReLU())
    

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    
    return model


def resizeBoxes(boxes,originSize,newSize):
    """
    resize the bounding boxes as well after image resize transformation

    Args:
        boxes (FloatTensor[N,4]): the coordinates of the N bounding boxes in [x1,y1,x2,y2].
                        x1,y1 refers to the top-left point of the bounding box
                        x2,y2 refers to the bottom-right of the bounding box

        originSize (2-tuple): (width,height) of the image before resizing when the bounding_boxes are created
        newSize ([type]): (width,height) of the image after resizing

    Returns:
        list: bounding boxes after resizing
    """

    ratioX=newSize[0]/originSize[0]
    #Resize the x-value
    boxes[:,(0,2)]*=ratioX


    ratioY=newSize[1]/originSize[1]
    #Resize the y-value
    boxes[:,(1,3)]*=ratioY

    return boxes



def draw(img,target,dataset="Labelbox",file=None):
    """
    draw the bounding boxes as well as with its category and confidence_score in an image

    Args:
        img (FloatTensor[c,h,w]): c represents number of channels, usually it's 3
                                    h represents the height of the image
                                    w represents the width of the image

        target (dict): its field includes:
                        boxes(list):each elements is [x1,y1,x2,y2]
                        labels(list): each element is an integer represents the category, ranges from 1~4, wich represents Door,Knob,Stairs,Ramp respectively
                        scores(list,optional):each element is a float, represents the confidence level for this prediction  
        dataset (str, optional): "Labelbox" or "Coco". Defaults to "Labelbox".
        file (str, optional): the file name of the image you want to save. Defaults to None.

    Raises:
        NameError: if the dataset name is not "Labelbox" or "Coco"
        NameError: if this system is not windows or linux
    """

    isTensor=torch.is_tensor(target["labels"]) #Verify if target[] is a tensor type or list type

    #Open the categories file
    with open("categories.json") as f:
        #It is a list contains dicts
        categories=json.load(f)[dataset]

    if dataset=="Coco":
        print("Showing images on Coco dataset")
        color={k["name"]:(random.randint(0,25)*10,random.randint(0,25)*10,
        random.randint(0,25)*10) for k in categories}
    
    elif dataset=="Labelbox":
        print("Showing images on Labelbox dataset")
        color={"None":(0,0,0),"Door":(255,0,0), 
        "Knob":(0,200,255),"Stairs":(0,255,0),"Ramp":(255,182,193)}
        
    else:
        print("Invalid dataset\n")
        raise NameError('The dataset name is wrong in draw()')
    
    
    #unpack target dict {"boxes":boxes,"labels":labels,......}
    boxes=target["boxes"]
    labels=target["labels"]
    if isTensor:
        boxes=boxes.tolist() #convert tensor to list
        labels=labels.tolist()

    labels=[categories[i]["name"] for i in labels]

    if "scores" in target: 
        scores=target["scores"]
        if isTensor:
            scores=scores.tolist()
        scores=[":"+str(int(s*100))+"%" for s in scores]
        print("Image visualization based on model's predictation")
    else:
        scores=[""]*len(labels)
        print("Image visualization based on ground truth")


    #Convert tensor[c,h,w] to PIL image
    transform =torchvision.transforms.ToPILImage(mode='RGB')
    img=img.cpu()
    img=transform(img)
    font_size=int(img.size[0]*12.0/800)
    if sys.platform.startswith('linux'):
        #Linux
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSerif.ttf",font_size)
    elif sys.platform.startswith('win'): 
        #Windows
       font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf",font_size)
    else:
        raise NameError("This system name can't be identified:"+sys.platform)

    draw=ImageDraw.Draw(img)

    for i in range(len(boxes)):

        #[x0,y0,x1,y1]
        draw.rectangle(boxes[i],outline=color[str(labels[i])],width=int(img.size[0]/500.0))
        draw.rectangle([boxes[i][0],boxes[i][1]-font_size-2,boxes[i][2],boxes[i][1]],
        fill=color[str(labels[i])]) 
        text=str(labels[i])+scores[i]

        draw.text((boxes[i][0]+1,boxes[i][1]-font_size-1),
            text=text,fill=(0,0,0),font=font) 
    

    if file!=None:
        img.save(file)
        print("image has been saved-"+file)

    plt.imshow(img)
    plt.axis('on')
    plt.show()


def getIou(boxA,boxB):
    """
    calculate the Intersection over Union(IoU) between two areas

    Args:
        boxA (list): the bounding boxes of the first area
        boxB (list): the bounding boxes of the second area

    Returns:
        float: the intersection over union 
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    
    boxAArea = (boxA[2] - boxA[0] ) * (boxA[3] - boxA[1] )
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

#boxes1:ground truth 
#boxes2:predictation
def getTp(boxes1,boxes2,THRESHOLD_IOU):
    """
    get the number of true positives

    Args:
        boxes1 (list): each element is in [x1,y1,x2,y2]
        boxes2 (list): each element is in [x1,y1,x2,y2]
        THRESHOLD_IOU (float): IoU threshold 

    Returns:
        integer: the truth positives
    """
    
    #Store the index of a ground truth if it's already corresponding to a predict label
    index=[]
    tp=0
    for box2 in boxes2: 
        for i in range(len(boxes1)):
            if getIou(boxes1[i],box2)>THRESHOLD_IOU and i not in index:
                index.append(i)
                tp+=1

    return tp


def predictOnImageFolder(img_folder,model,THRESHOLD_IOU=0,dataset="Labelbox",NMS=False):
    """
    draw prediction on all the images in the image folder

    Args:
        img_folder (str): the folder path where all images are stored
        model (pytorch Faster RCNN ): 
        THRESHOLD_IOU (int, optional): set the IoU threshold for NMS algorithm. Defaults to 0.
        dataset (str, optional): the name of the dataset. Accepted value is "Labelbox" or "Coco". Defaults to "Labelbox".
        NMS (bool, optional): whether to apply non-maximum suppresion(NMS). Defaults to False.
    """

    model.eval()
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()]) 

    
    name=os.listdir(img_folder)
    for n in name:
        if not n[-7:-4]=="nms" or not n[-7:-4]=="ict":
            img_path=os.path.join(img_folder,n)
            img=Image.open(img_path).convert("RGB")
            x=transform(img)
            x=x.unsqueeze(0)
            x=x.cuda()
            target=model(x)

            #Using NMS
            if NMS:
                score=target[0]["scores"].tolist()
                label=target[0]["labels"].tolist()
                box=target[0]["boxes"].tolist()
                target[0]["boxes"],target[0]["labels"],target[0]["scores"]=nms(box,label,score,THRESHOLD_IOU)
            

            #image's file name prefix
            postfix="_predict.jpg"
            #put x back to cpu
            x=x.cpu()

            draw(x[0],target[0],"Labelbox",file=os.path.join(img_folder,n[:-4]+postfix))

def toNumpy(x):
    """
    change input into numpy array

    Args:
        x (torch.tensor or list): 

    Returns:
        nd.array: 
    """
    typeClass=str(type(x))
    #If x is a Torch.tensor
    if "torch" in typeClass:
        x=x.detach().cpu().numpy()

    #If x is a list
    elif "list" in typeClass:
        x=np.asarray(x)
        

    return x

def toList(x):
    """
    change input into list  

    Args:
        x (nd.array or torch.tensor):
    Returns:
        list:
    """
    typeClass=str(type(x))

    #If x is a Torch.tensor
    if "torch" in typeClass:
        x=x.tolist()

    #If x is a numpy array
    elif "numpy" in typeClass:
        x=x.tolist()

    return x


def toTensor(x):
    """
    change input into torch.tensor format

    Args:
        x (list or nd.array): 

    Returns:
        torch.tensor: 
    """
    typeClass=str(type(x))

    #If x is a list
    if "list" in typeClass:
        x=torch.Tensor(x)

    #If x is a numpy array
    elif "numpy" in typeClass:
        x=torch.from_numpy(x)

        #Manually change to float32 if it is float64
        if "float" in str(x.dtype):
            x=x.type(torch.float32)

    return x

