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
from decode import *
import os
import wandb
import copy
import statistics
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
                target[0]["boxes"],target[0]["labels"],target[0]["scores"]=nms(box,label,score,IoUThreshold)
            

            #image's file name prefix
            postfix="_predict.jpg"
            #put x back to cpu
            x=x.cpu()

            draw(x[0],target[0],"Labelbox",file=os.path.join(img_folder,n[:-4]+postfix))





def nms(bounding_boxes,label,confidence_score,threshold):
    """
    Non-max Suppression Algorithm 
    @param list  Object candidate bounding boxes
    @param list label 
    @param list  Confidence score of bounding boxes
    @param float IoU threshold
    @return list picked_boxes,picked_label,picked_score
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

    return picked_boxes,picked_label,picked_score

def snms(bounding_boxes,label,confidence_score,threshold):
    """
    Non-max Suppression Algorithm 
    @param list  Object candidate bounding boxes 
    @param list  Confidence score of bounding boxes
    @param float IoU threshold
    @return list picked_boxes,picked_label,picked_score
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

    return picked_boxes,picked_label,picked_score


def Filter(bounding_boxes,label,confidence_score):
    """
    @param list  Object candidate bounding boxes 
    @param label predicted category
    @param list  Confidence score of bounding boxes
    @return list bounding_boxes,label,confidence_score
    """

    # Bounding boxes
    bounding_boxes = toNumpy(bounding_boxes)
    # Confidence scores of bounding boxes
    confidence_score = toNumpy(confidence_score)
    #label
    label=toNumpy(label)


    #Sort all labels based on score descending 
    index=(-confidence_score).argsort()
    bounding_boxes=bounding_boxes[index]
    label=label[index]
    confidence_score=confidence_score[index]



    #Filter out windows that are above door
    door_index=np.where(label==1)[0]
    door_score_largest=np.argmax(confidence_score[door_index])

    y_top=bounding_boxes[door_index[0]][1]
    y_bottom=bounding_boxes[door_index[0]][3]
    door_filter_index=np.where(np.logical_or(bounding_boxes[door_index,3]<=y_top,bounding_boxes[door_index,1]>=y_bottom))
    door_filter_index=door_filter_index[0]

    #Delete these labels
    bounding_boxes=np.delete(bounding_boxes,door_index[door_filter_index],axis=0)
    label=np.delete(label,door_index[door_filter_index])
    confidence_score=np.delete(confidence_score,door_index[door_filter_index])


    #Filter out Knob based on door. No two knob in the same door. No knob go beyond door.
    door_index=np.where(label==1)[0]
    knob_index=np.where(label==2)[0]

    x_middle=(bounding_boxes[knob_index,0]+bounding_boxes[knob_index,2])/2.0
    y_middle=(bounding_boxes[knob_index,1]+bounding_boxes[knob_index,3])/2.0

    knob_keep_index=[]
    for i in door_index:
        x1=bounding_boxes[i][0]<x_middle
        x2=bounding_boxes[i][2]>x_middle
        y1=bounding_boxes[i][1]<y_middle
        y2=bounding_boxes[i][3]>y_middle
        xy=np.array((x1,x2,y1,y2))

        knob_keep_id=np.where(np.logical_and.reduce(xy))[0]

        if knob_keep_id.size>0:
            
            knob_keep_index.append(knob_keep_id[0])

    knob_filter_index=[i for i in range(len(knob_index)) if i not in knob_keep_index]
    #Delete these labels
    bounding_boxes=np.delete(bounding_boxes,knob_index[knob_filter_index],axis=0)
    label=np.delete(label,knob_index[knob_filter_index],axis=0)
    confidence_score=np.delete(confidence_score,knob_index[knob_filter_index],axis=0)


    return toList(bounding_boxes),toList(label),toList(confidence_score)


# #Test cases
# boxes=np.array([[0,30,10,50],[0,40,1,41],[5,45,6,46],[15,40,16,41],[20,10,30,20],[20,40,30,60]])
# labels=np.array([1,2,2,2,1,1,1])
# scores=np.array([0.9,0.3,0.6,0.9,0.3,0.5])
# box,label,score=filter(boxes,labels,scores)
# print("boxes:",box,"\nlabels:",label,"\nscore:",score)

def filterDoor(bounding_boxes,label,confidence_score,new_size,image_id,folder):
    """
    @param list  Object candidate bounding boxes 
    @param label predicted category
    @param list  Confidence score of bounding boxes
    @new_size tuple the dimension of the image
    @image_id the file name of this image
    @folder a path to where pointCloud is stored
    @return list bounding_boxes,label,confidence_score
    """
    
    z_threshold=0.3
    z_sidewalk=2.4

    # Bounding boxes
    bounding_boxes = toNumpy(bounding_boxes)
    # Confidence scores of bounding boxes
    confidence_score = toNumpy(confidence_score)
    #label
    label=toNumpy(label)
    
    #Delete door labels whose scores are lower than SCORE_THRESHOLD
    # delete_index=np.where(np.logical_and(label==1,confidence_score<SCORE_THRESHOLD))
    # bounding_boxes=np.delete(bounding_boxes,
    # delete_index,axis=0)
    # label=np.delete(label,delete_index)
    # confidence_score=np.delete(confidence_score,delete_index)

    pointCloud=np.load(os.path.join(folder,image_id[:-6]+".npy"))
    plane=np.load(os.path.join(folder,image_id[:-6]+"_plane.npy"))

    #Sort all labels based on score descending 
    index=(-confidence_score).argsort()
    bounding_boxes=bounding_boxes[index]
    label=label[index]
    confidence_score=confidence_score[index]
    # print("bounding_boxes:",bounding_boxes)
    #print("scores:",confidence_score)
    #print("label:",label,label.shape)

    #Filter out windows that are above door
    door_index=np.where(label==1)[0]
    door_score_largest=np.argmax(confidence_score[door_index])  

    x_left=bounding_boxes[door_index[0],0]
    x_right=bounding_boxes[door_index[0],2]
    y_bottom=bounding_boxes[door_index[0],3]


    #Make a threshold here and take out some doors to verify
    z0=min(getZ(x_left,y_bottom,pointCloud,new_size,image_id),getZ(x_right,y_bottom,pointCloud,new_size,image_id))

    z=np.minimum(getZ(bounding_boxes[door_index,0],bounding_boxes[door_index,3],pointCloud,new_size,image_id),
    getZ(bounding_boxes[door_index,2],bounding_boxes[door_index,3],pointCloud,new_size,image_id))
    # print("z:",np.minimum(getZ(bounding_boxes[:,0],bounding_boxes[:,3],pointCloud,new_size,image_id),getZ(bounding_boxes[:,2],bounding_boxes[:,3],pointCloud,new_size,image_id)))
    door_filter_index_top=np.where(z<z0-z_threshold)[0]
    #print("top:",door_filter_index_top)
    # door_filter_index_bot=np.where(np.logical_and(z>z0-z_threshold,z!=z0,z<z_sidewalk-z_threshold))[0]
    door_filter_index_bot=np.where(z>z0-z_threshold)[0]
    # print("bot:",door_filter_index_bot)
    # print("filter:",door_filter_index_top)
    # print("door index:",door_index)
    x1=bounding_boxes[door_index[door_filter_index_top],0]
    y1=bounding_boxes[door_index[door_filter_index_top],3]
    

    x2,y2=bounding_boxes[door_index[door_filter_index_top],2],y_bottom

    sr_index=np.where(np.logical_or(label==3,label==4))
   
    sr_x1=bounding_boxes[sr_index,0]
    sr_x2=bounding_boxes[sr_index,2]
    sr_y1=bounding_boxes[sr_index,1]

    #x1-(x2-x1)*1/3<sr_x1<sr_x1<x1+(x2-x1)*1/3
    boolean_x1=np.ones((x1.size,sr_x1.size))
    srx1=boolean_x1*sr_x1
    srx1=srx1.T
    boolean_x2=np.ones((x2.size,sr_x2.size))
    srx2=boolean_x2*sr_x2
    srx2=srx2.T

    boolean_x=np.logical_and(srx1<=x2,srx2>=x1)


    # print("sr_x:",sr_x)
    # print("x1:",x1)
    # print("bolean_x:",boolean_x)

    #z+threshold<sr_z1<z1+z_threshold
    z1=z[door_filter_index_top]
    sr_z=getZ(sr_x1,sr_y1,pointCloud,new_size,image_id)
    boolean_z=np.ones((z1.size,sr_z.size))
    sr_z=boolean_z*sr_z
    sr_z=sr_z.T
    
    boolean_z=np.logical_and(sr_z<z1+z_threshold,sr_z>z1-z_threshold)
    # print("sr_z:",sr_z)
    # print("z1:",z1)
    # print("bolean_z:",boolean_z)

    #boolean and between (boolean of x and y)
    boolean_door=np.logical_and(boolean_x,boolean_z)

    #boolean or in each column
    #print(door_filter_index_top,boolean_door)
    boolean_door=np.sum(boolean_door,axis=0)
    door_filter_index_top=door_filter_index_top[np.where(boolean_door==0)[0]]
    # print("door filter index:",door_filter_index_top)


    #Deal with door_filter_bot
    # x1=bounding_boxes[door_index[door_filter_index_bot],0]
    # y1=bounding_boxes[door_index[door_filter_index_bot],3]
    
    # x2,y2=bounding_boxes[door_index[door_filter_index_bot],2],y_bottom

    # sr_index=np.where(np.logical_or(label==3,label==4))
   
    # sr_x1=bounding_boxes[sr_index,0]
    # sr_y1=bounding_boxes[sr_index,1]

    # #x1-(x2-x1)*1/3<sr_x1<sr_x1<x1+(x2-x1)*1/3
    # boolean_x=np.ones((x1.size,sr_x1.size))
    # sr_x=boolean_x*sr_x1
    # sr_x=sr_x.T
    # boolean_x=np.logical_and(x1-(x2-x1)*1/3<sr_x,sr_x<x1+(x2-x1)*1/3)
    # # print("sr_x:",sr_x)
    # # print("x1:",x1)
    # # print("bolean_x:",boolean_x)

    # #z+threshold<sr_z1<z1+z_threshold
    # z1=z[door_filter_index_bot]
    # sr_z=getZ(sr_x1,sr_y1,pointCloud,new_size,image_id)
    # boolean_z=np.ones((z1.size,sr_z.size))
    # sr_z=boolean_z*sr_z
    # sr_z=sr_z.T
    
    # boolean_z=np.logical_and(sr_z<z1+z_threshold,sr_z>z1-z_threshold)
    # # print("sr_z:",sr_z)
    # # print("z1:",z1)
    # # print("bolean_z:",boolean_z)

    # #boolean and between (boolean of x and y)
    # boolean_door=np.logical_and(boolean_x,boolean_z)

    # #boolean or in each column
    # #print(door_filter_index_bot,boolean_door)
    # boolean_door=np.sum(boolean_door,axis=0)
    # door_filter_index_bot=door_filter_index_bot[np.where(boolean_door==0)[0]]

    # #Applied plane to filter any doors that are lower than the door with the highest score
    
    # x1=bounding_boxes[door_index[door_filter_index_bot],0]
    # y1=bounding_boxes[door_index[door_filter_index_bot],1]
    # x2=bounding_boxes[door_index[door_filter_index_bot],2]
    # y2=bounding_boxes[door_index[door_filter_index_bot],3]

    # x1=(x2-x1)*1/3+x1
    # x1=x1.astype(int)

    # x2=(x2-x1)*2/3+x1
    # x2=x2.astype(int)

    # y1=(y2-y1)*1/3+y1
    # x1=x1.astype(int)

    # y2=(y2-y1)*1/2+y2
    # x1=x1.astype(int)

    # x=np.linspace(x1,x2,10)
    # y=np.linspace(y1,y2,10)

    # #print("x1:",x1.shape)
    
    # x=getZ(x,y,pointCloud,new_size,image_id)
    #print(x.shape,"\n",x,"\n\n")
    # for j in range(x.shape[1]):
    #     l=[]
    #     for i in range(x.shape[0]):
            
    #         if x[i,j]!=1.00000000e+19:
    #             l.append(x[i,j])
            

    #     print("std:",statistics.stdev(l))

    
    # std=np.std(x,axis=0)
    # # print("\nstd:",std)
    # print("index:",door_index[door_filter_index_bot],"\nscores:",confidence_score[door_index[door_filter_index_bot]])


    #Delete these labels
    bounding_boxes=np.delete(bounding_boxes,
    door_index[door_filter_index_top],axis=0)
    label=np.delete(label,door_index[door_filter_index_top])
    # print(label.shape)
    confidence_score=np.delete(confidence_score,door_index[door_filter_index_top])

    # bounding_boxes=np.delete(bounding_boxes,
    # door_index[np.concatenate((door_filter_index_top,door_filter_index_bot))],axis=0)
    # label=np.delete(label,door_index[np.concatenate((door_filter_index_top,door_filter_index_bot))])
    # confidence_score=np.delete(confidence_score,door_index[np.concatenate((door_filter_index_top,door_filter_index_bot))])

    return toList(bounding_boxes),toList(label),toList(confidence_score)


# bounding_boxes=[[50,50,70,70],[10,40,30,60],[10,53,25,70],[80,40,100,60],[85,62,100,80]]
# label=[1,1,4,1,3]
# confidence_score=[0.8,0.4,0.4,0.3,0.3]
# boxes,label,score=filterDoor(bounding_boxes,label,confidence_score)
# print(boxes,"\n\n",label,"\n\n",score)

def getZ(x,y,pointCloud,new_size,image_id):
    #Mapping from cropped_image to panorama
    pano_x,pano_y=16384,8192
    pointCloud_x,pointCloud_y=512,256
    img_width,img_height=3584,2560
    width_left_offset=2048
    width_right_offset=10752
    height_offset=3072

    y=(y/new_size[1]*img_height+height_offset)*pointCloud_y/pano_y
    if image_id[-5]=="0":
        x=(x/new_size[0]*img_width+width_left_offset)*pointCloud_x/pano_x

    elif image_id[-5]=="1":
        x=(x/new_size[0]*img_width+width_right_offset)*pointCloud_x/pano_x
    
        
    x=x.astype(int)
    y=y.astype(int)


    #Return z-axis value
    return pointCloud[y,x,2]

def getLatLon(x,y,folder,new_size,image_id):

    latlon = findLatLon(os.path.join(folder,image_id[:-6] + ".xml"))
    pointCloud=np.load(os.path.join(folder,image_id[:-6]+".npy"))

    clat = latlon[0]
    clon = latlon[1]
    yaw = latlon[2]

    if(yaw > 180):
        yaw = yaw - 180
    else:
        yaw = 180 + yaw
        
    #Convert to numpy
    x=toNumpy(x)
    y=toNumpy(y)

    #Mapping from cropped_image to panorama
    pano_x,pano_y=16384,8192
    pointCloud_x,pointCloud_y=512,256
    img_width,img_height=3584,2560
    width_left_offset=2048
    width_right_offset=10752
    height_offset=3072

    y=(y/new_size[1]*img_height+height_offset)*pointCloud_y/pano_y
    if image_id[-5]=="0":
        x=(x/new_size[0]*img_width+width_left_offset)*pointCloud_x/pano_x

    elif image_id[-5]=="1":
        x=(x/new_size[0]*img_width+width_right_offset)*pointCloud_x/pano_x

    else:
        raise AttributeError

    x=x.astype(int)
    y=y.astype(int)

   
    #Calculate lat lon
    dx = pointCloud[y,x,0]
    dy = pointCloud[y,x,1]

    
    index_null=np.logical_or(dx>999,dy>999)[0]

    rdx = dx*np.cos(np.radians(yaw)) + dy*np.sin(np.radians(yaw))
    rdy = -1*dx*np.sin(np.radians(yaw)) + dy*np.cos(np.radians(yaw))
    
    
    dlat = rdy / 111111
    dlon = rdx / (111111 * np.cos(np.radians(clat)))
    
    

    lat = dlat + clat
    lon = dlon + clon

    
    #Turn into 0 at null index
    lat[index_null]=0
    lon[index_null]=0

    return lat,lon

def getPlane(x,y,plane,new_size,image_id):
    
    #Mapping from cropped_image to panorama
    pano_x,pano_y=16384,8192
    pointCloud_x,pointCloud_y=512,256
    img_width,img_height=3584,2560
    width_left_offset=2048
    width_right_offset=10752
    height_offset=3072

    y=(y/new_size[1]*img_height+height_offset)*pointCloud_y/pano_y
    if image_id[-5]=="0":
        x=(x/new_size[0]*img_width+width_left_offset)*pointCloud_x/pano_x

    elif image_id[-5]=="1":
        x=(x/new_size[0]*img_width+width_right_offset)*pointCloud_x/pano_x


    if type(x)==int:
        x=int(x)
        y=int(y)
    else:
        x=x.astype(int)
        y=y.astype(int)

    
    #Return plane indics
    return plane[y,x]

def findLatLon(path_to_metadata_xml):
    pano = {}
    pano_xml = open(path_to_metadata_xml, 'rb')
    tree = ET.parse(pano_xml)
    root = tree.getroot()
    for child in root:
        if child.tag == 'projection_properties':
            pano[child.tag] = child.attrib
        if child.tag == 'data_properties':
            pano[child.tag] = child.attrib
    
    return (float(pano['data_properties']['lat']), float(pano['data_properties']['lng']), float(pano['projection_properties']['pano_yaw_deg']))

def removeDoors(box, label,score,image_id,folder):
    #Read the planes data
    depthMap = getDepthMap(os.path.join(folder,image_id[:-6] + ".xml"))
    depthMapData = parse(depthMap)
    header = parseHeader(depthMapData)
    data = parsePlanes(header, depthMapData)
    planes = data["indices"]
    
    #Deep copy
    box_copy=copy.deepcopy(box)

    for ar in box:
        ar[0] = ar[0] * 3584 / 1000
        ar[2] = ar[2] * 3584 / 1000
        ar[1] = ar[1] * 2560 / 1000
        ar[3] = ar[3] * 2560 / 1000
    for ar in box:
        ar[0] = int(ar[0] / 32) + 64
        ar[2] = int(ar[2] / 32) + 64
        if image_id[-5]=="1":
            ar[0] += 272
            ar[2] += 272
        ar[1] = int(ar[1] / 32) + 96
        ar[3] = int(ar[3] / 32) + 96
    i = 0
    while i < len(box):
        if label[i] == 1:
            ar = box[i]
            tl = ar[1] * 512 + ar[0]
            tr = ar[1] * 512 + ar[2]
            bl = ar[3] * 512 + ar[0]
            br = ar[3] * 512 + ar[2]
            if planes[tl] != planes[tr]:
                print("ERROR")
            if planes[tl] != planes[bl] or planes[tr] != planes[br]:
                i+=1
                continue
            remove = True
            for j in range(len(box)):
                if label[j] == 3:
                    stair = box[j]
                    if not (stair[0] > ar[2] + 3 or stair[2] + 3 < ar[0] or stair[1] > ar[3] + 5 or stair[3] + 5 < ar[1]):
                        bls = (stair[3] + 5) * 512 + stair[0]
                        brs = (stair[3] + 5) * 512 + stair[2]
                        if planes[tl] != planes[bls] or planes[tr] != planes[brs]:
                            remove = False
            if remove:
                box_copy.pop(i)
                box.pop(i)
                label.pop(i)
                score.pop(i)
                i-=1
        i+=1
    
    return box_copy,label,score

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

