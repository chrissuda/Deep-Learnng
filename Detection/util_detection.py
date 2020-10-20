



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
import wandb


def transfer(model,num_classes):
    '''
    Change the Faster RCNN model to adapt to our training category
    @param num_classes: background+classes
    '''

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
    '''
    resize the bounding box because of data transformation
    @param boxes [x1,y1,x2,y2]
    @param originSize (x,y)
    @param newSize (x,y)
    @return boxes [x1,y1,x2,y2]
    '''

    ratioX=newSize[0]/originSize[0]
    boxes[:,0::2]*=ratioX

    ratioY=newSize[1]/originSize[1]
    boxes[:,1::2]*=ratioY
    return boxes;



def draw(img,target,dataset="Labelbox",file=None):
    '''
    @param img a tensor[c,h,w]
    @param target a dict contains various boxes,labels
    @param dataset a String to indicate whether it is "Coco" or "Labelbox"
    @param file the name of the image you want to save
    '''

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

    try: 
        scores=target["scores"]
        if isTensor:
            scores=scores.tolist()
        scores=[":"+str(int(s*100))+"%" for s in scores]
        print("Image visualization based on model's predictation")
    except:
        scores=[""]*len(labels)
        print("Image visualization based on ground truth")


    #Convert tensor[c,h,w] to PIL image
    transform =torchvision.transforms.ToPILImage(mode='RGB')
    img=img.cpu()
    img=transform(img)
    font_size=int(img.size[0]*12.0/800)
    try:
        #Linux
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSerif.ttf",font_size)
    except:
        #Windows
       font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf",font_size)
    
    draw=ImageDraw.Draw(img)

    for i in range(len(boxes)):
        
        #[x0,y0,x1,y1]
        draw.rectangle(boxes[i],outline=color[str(labels[i])],width=2) 
        text=str(labels[i])+scores[i]

        draw.text((boxes[i][0],boxes[i][1]+1),
            text=text,fill=color[str(labels[i])],font=font) 
    

    if file!=None:
        img.save(file)
        print("image has been saved-"+file)

    plt.imshow(img)
    plt.axis('on')
    plt.show()


def IoU(boxA,boxB):
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
def checkTp(boxes1,boxes2,threshold):
    #Store the index of a groudn truth if it's already corresponding to a predict label
    index=[]

    tp=0
    for box2 in boxes2: 
        for i in range(len(boxes1)):
            if IoU(boxes1[i],box2)>threshold and i not in index:
                index.append(i)
                tp+=1

    return tp


def predictOnImageFolder(img_folder,model,IoUThreshold=0,dataset="Labelbox",NMS=False):
    '''
    Predict labels on all the images in a specific folder and annotated them
    @param img_folder
    @param model
    @param IoUThreshold
    @param dataset A string: "Coco" or "Labelbox"
    '''

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
                target[0]["scores"],target[0]["labels"],target[0]["boxes"]=nms(box,label,score,IoUThreshold)

                #image's file name prefix. Indicating nms or not
                postfix="_predict_nms.jpg"
            
            else:
                #image's file name prefix
                postfix="_predict.jpg"
            #put x back to cpu
            x=x.cpu()

            draw(x[0],target[0],"Labelbox",file=os.path.join(img_folder,n[:-4]+postfix))





def nms(bounding_boxes,label,confidence_score,threshold):
    """
    Non-max Suppression Algorithm 
    @param list  Object candidate bounding boxes 
    @param list  Confidence score of bounding boxes
    @param float IoU threshold
    @return list picked_score,picked_label,picked_boxes
    """

    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], [],[]

    # Bounding boxes
    boxes = toNumpy(bounding_boxes)
    # Confidence scores of bounding boxes
    score = toNumpy(confidence_score)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]



    # Picked bounding boxes
    picked_boxes = []
    picked_score = []
    picked_label= []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 0.1) * (end_y - start_y + 0.1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(toList(boxes[index]))
        picked_score.append(toList(score[index]))
        picked_label.append(toList(label[index]))

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 0.1)
        h = np.maximum(0.0, y2 - y1 + 0.1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_score,picked_label,picked_boxes

def snms(bounding_boxes,label,confidence_score,threshold):
    """
    Non-max Suppression Algorithm 
    @param list  Object candidate bounding boxes 
    @param list  Confidence score of bounding boxes
    @param float IoU threshold
    @return list picked_score,picked_label,picked_boxes
    """

    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], [],[]

    # Bounding boxes
    boxes = toNumpy(bounding_boxes)
    # Confidence scores of bounding boxes
    score = toNumpy(confidence_score)
    #label
    label=toNumpy(label)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []
    picked_label= []

    for i in range(1,5):
        labelIndex=np.where(label==i)
        b=boxes[labelIndex]#bounding boxes
        s=score[labelIndex]#score
        l=label[labelIndex]#label

        # coordinates of bounding b
        start_x = b[:, 0]
        start_y = b[:, 1]
        end_x = b[:, 2]
        end_y = b[:, 3]


        # Compute areas of bounding boxes
        areas = (end_x - start_x + 1) * (end_y - start_y + 1)

        # Sort by confidence score of bounding boxes
        order = np.argsort(s)

        # Iterate bounding boxes
        while order.size > 0:
            # The index of largest confidence score
            index = order[-1]
            
            # Pick the bounding box with largest confidence score
            picked_boxes.append(toList(b[index]))
            picked_score.append(toList(s[index]))
            picked_label.append(toList(l[index]))

            # Compute ordinates of intersection-over-union(IOU)
            x1 = np.maximum(start_x[index], start_x[order[:-1]])
            x2 = np.minimum(end_x[index], end_x[order[:-1]])
            y1 = np.maximum(start_y[index], start_y[order[:-1]])
            y2 = np.minimum(end_y[index], end_y[order[:-1]])

            # Compute areas of intersection-over-union
            w = np.maximum(0.0, x2 - x1 + 1)
            h = np.maximum(0.0, y2 - y1 + 1)
            intersection = w * h

            # Compute the ratio between intersection and union
            ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

            left = np.where(ratio < threshold)
            order = order[left]

    return picked_score,picked_label,picked_boxes


def toNumpy(x):

    typeClass=str(type(x))
    #If x is a Torch.tensor
    if "torch" in typeClass:
        x=x.detach().cpu().numpy()

    #If x is a list
    elif "list" in typeClass:
        x=np.asarray(x)
        

    return x

def toList(x):
    typeClass=str(type(x))

    #If x is a Torch.tensor
    if "torch" in typeClass:
        x=x.tolist()

    #If x is a numpy array
    elif "numpy" in typeClass:
        x=x.tolist()

    return x


def toTensor(x):
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