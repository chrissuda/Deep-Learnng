'''
This file provides several functions to filter labels
'''

import numpy as np
import json
import matplotlib.pyplot as plt
import random
import time
from util_detection import *
from util_geolocation import *
import os

def nms(boxes,labels,scores,THRESHOLD_IOU):
    """
    Non-maximum Suppression Algorithm

    Args:
        boxes(list): a list of bounding box output by Faster RCNN model for an image
                            each individual box is in this format: [x1,y1,x2,y2]
        
        labels(list): a list of label output by Faster RCNN model for an image,
                    each label is an integer, ranges from 1-4. 
        
        scores(list): a list of score output by Faster RCNN model for an image,
                                each score represent the confidence level for predicting this label

        THRESHOLD_IOU(float): one label with a smaller confidence score will get eliminated 
                        if the IoU between these two labels is larget than threshold.

    Returns:
        list: remaining bounding boxes after eliminating some labels
        list: remaining label after eliminating some labels
        list: remaining score after eliminating some labels

    """

    # If no bounding boxes, return empty list
    if len(boxes) == 0:
        return [], [],[]

    # Bounding boxes
    boxes = toNumpy(boxes)
    # Confidence scores of bounding boxes
    scores = toNumpy(scores)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]



    # Picked bounding boxes
    picked_boxes = []
    picked_scores = []
    picked_labels= []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 0.1) * (end_y - start_y + 0.1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(scores)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(toList(boxes[index]))
        picked_scores.append(toList(scores[index]))
        picked_labels.append(toList(labels[index]))

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

        left = np.where(ratio < THRESHOLD_IOU)
        order = order[left]

    return picked_boxes,picked_labels,picked_scores

def snms(boxes,labels,scores,THRESHOLD_IOU):
     
    """
     perform nms on labels in the same category only

    Args:
        boxes(list): a list of bounding box output by Faster RCNN model for an image
                            each individual box is in this format: [x1,y1,x2,y2]
        
        labels(list): a list of label output by Faster RCNN model for an image,
                    each label is an integer, ranges from 1-4. 
        
        scores(list): a list of score output by Faster RCNN model for an image,
                                each score represent the confidence level for predicting this label

        THRESHOLD_IOU(float): one label with a smaller confidence score will get eliminated 
                        if the IoU between these two labels is larget than threshold.

    Returns:
        list: remaining bounding boxes after eliminating some labels
        list: remaining label after eliminating some labels
        list: remaining score after eliminating some labels

    """

    # If no bounding boxes, return empty list
    if len(boxes) == 0:
        return [], [],[]

    # Bounding boxes
    boxes = toNumpy(boxes)
    # Confidence scores of bounding boxes
    scores = toNumpy(scores)
    #label
    labels=toNumpy(labels)

    # Picked bounding boxes
    picked_boxes = []
    picked_scores = []
    picked_labels= []

    for i in range(1,5):
        labelIndex=np.where(labels==i)
        b=boxes[labelIndex]#bounding boxes
        s=scores[labelIndex]#score
        l=labels[labelIndex]#label

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
            picked_scores.append(toList(s[index]))
            picked_labels.append(toList(l[index]))

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

            left = np.where(ratio < THRESHOLD_IOU)
            order = order[left]

    return picked_boxes,picked_labels,picked_scores

def filterDoor(boxes,labels,scores,new_size,image_id,folder):
    """
    filtering doors using point cloud information

    Args:
        boxes(list): a list of bounding box output by Faster RCNN model for an image
                            each individual box is in this format: [x1,y1,x2,y2]
        
        labels(list): a list of label output by Faster RCNN model for an image,
                    each label is an integer, ranges from 1-4. 
        
        scores(list): a list of score output by Faster RCNN model for an image,
                                each score represent the confidence level for predicting this label

        new_size (2-tuple): the sizes of an image after resizing before feeding into the model, 
                            in (width,height)

        image_id (str): an unique identifer of an image in the dataset.

        folder (str): a path to the folder where your xml files and images are stored.

    Returns:
        list: remaining bounding boxes after eliminating some labels
        list: remaining label after eliminating some labels
        list: remaining score after eliminating some labels

    """    
    
    z_threshold=0.3

    # Bounding boxes
    boxes = toNumpy(boxes)
    # Confidence scores of bounding boxes
    scores = toNumpy(scores)
    #label
    labels=toNumpy(labels)
    
    #Delete door labels whose scores are lower than SCORE_THRESHOLD
    
    pointCloud=np.load(os.path.join(folder,image_id[:-6]+".npy"))

    #Sort all labels based on score descending 
    index=(-scores).argsort()
    boxes=boxes[index]
    labels=labels[index]
    scores=scores[index]
    # print("bounding_boxes:",bounding_boxes)
    #print("scores:",confidence_score)
    #print("label:",label,label.shape)

    #Filter out windows that are above door
    door_index=np.where(labels==1)[0]
    #If there are no doors here
    if door_index.size==0:
        return toList(boxes),toList(labels),toList(scores)

    door_score_largest=np.argmax(scores[door_index])  

    x_left=boxes[door_index[0],0]
    x_right=boxes[door_index[0],2]
    y_bottom=boxes[door_index[0],3]


    #Make a threshold here and take out some doors to verify
    z0=min(getZ(x_left,y_bottom,pointCloud,new_size,image_id),getZ(x_right,y_bottom,pointCloud,new_size,image_id))

    z=np.minimum(getZ(boxes[door_index,0],boxes[door_index,3],pointCloud,new_size,image_id),
    getZ(boxes[door_index,2],boxes[door_index,3],pointCloud,new_size,image_id))
    # print("z:",np.minimum(getZ(bounding_boxes[:,0],bounding_boxes[:,3],pointCloud,new_size,image_id),getZ(bounding_boxes[:,2],bounding_boxes[:,3],pointCloud,new_size,image_id)))
    door_filter_index_top=np.where(z<z0-z_threshold)[0]
    #print("top:",door_filter_index_top)
    # door_filter_index_bot=np.where(np.logical_and(z>z0-z_threshold,z!=z0,z<z_sidewalk-z_threshold))[0]
    door_filter_index_bot=np.where(z>z0-z_threshold)[0]
    # print("bot:",door_filter_index_bot)
    # print("filter:",door_filter_index_top)
    # print("door index:",door_index)
    x1=boxes[door_index[door_filter_index_top],0]
    y1=boxes[door_index[door_filter_index_top],3]
    

    x2,y2=boxes[door_index[door_filter_index_top],2],y_bottom

    sr_index=np.where(np.logical_or(labels==3,labels==4))
   
    sr_x1=boxes[sr_index,0]
    sr_x2=boxes[sr_index,2]
    sr_y1=boxes[sr_index,1]

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
    boxes=np.delete(boxes,
    door_index[door_filter_index_top],axis=0)
    labels=np.delete(labels,door_index[door_filter_index_top])
    # print(label.shape)
    scores=np.delete(scores,door_index[door_filter_index_top])

    # bounding_boxes=np.delete(bounding_boxes,
    # door_index[np.concatenate((door_filter_index_top,door_filter_index_bot))],axis=0)
    # label=np.delete(label,door_index[np.concatenate((door_filter_index_top,door_filter_index_bot))])
    # confidence_score=np.delete(confidence_score,door_index[np.concatenate((door_filter_index_top,door_filter_index_bot))])

    return toList(boxes),toList(labels),toList(scores)


def filterKnob(boxes,labels,scores):
    """
    filter knobs

    Args:
        boxes(list): a list of bounding box output by Faster RCNN model for an image
                            each individual box is in this format: [x1,y1,x2,y2]
        
        labels(list): a list of label output by Faster RCNN model for an image,
                    each label is an integer, ranges from 1-4. 
        
        scores(list): a list of score output by Faster RCNN model for an image,
                                each score represent the confidence level for predicting this label

    Returns:
        list: remaining bounding boxes after eliminating some labels
        list: remaining label after eliminating some labels
        list: remaining score after eliminating some labels

    """

    # Bounding boxes
    boxes = toNumpy(boxes)
    # Confidence scores of bounding boxes
    scores = toNumpy(scores)
    #label
    labels=toNumpy(labels)

    #Sort all labels based on score descending 
    index=(-scores).argsort()
    boxes=boxes[index]
    labels=labels[index]
    scores=scores[index]


    #Filter out Knob based on door. No two knobs can be in the same door. No knob can go beyond door.
    door_index=np.where(labels==1)[0]
    knob_index=np.where(labels==2)[0]

    x_middle=(boxes[knob_index,0]+boxes[knob_index,2])/2.0
    y_middle=(boxes[knob_index,1]+boxes[knob_index,3])/2.0

    knob_keep_index=[]
    for i in door_index:
        x1=boxes[i][0]<x_middle
        x2=boxes[i][2]>x_middle
        y1=boxes[i][1]<y_middle
        y2=boxes[i][3]>y_middle
        xy=np.array((x1,x2,y1,y2))

        knob_keep_id=np.where(np.logical_and.reduce(xy))[0]

        if knob_keep_id.size>0:
            
            knob_keep_index.append(knob_keep_id[0])

    knob_filter_index=[i for i in range(len(knob_index)) if i not in knob_keep_index]
    #Delete these labels
    boxes=np.delete(boxes,knob_index[knob_filter_index],axis=0)
    labels=np.delete(labels,knob_index[knob_filter_index],axis=0)
    scores=np.delete(scores,knob_index[knob_filter_index],axis=0)


    return toList(boxes),toList(labels),toList(scores)


# #Test cases
# boxes=np.array([[0,30,10,50],[0,40,1,41],[5,45,6,46],[15,40,16,41],[20,10,30,20],[20,40,30,60]])
# labels=np.array([1,2,2,2,1,1,1])
# scores=np.array([0.9,0.3,0.6,0.9,0.3,0.5])
# box,label,score=filter(boxes,labels,scores)
# print("boxes:",box,"\nlabels:",label,"\nscore:",score)




# bounding_boxes=[[50,50,70,70],[10,40,30,60],[10,53,25,70],[80,40,100,60],[85,62,100,80]]
# label=[1,1,4,1,3]
# confidence_score=[0.8,0.4,0.4,0.3,0.3]
# boxes,label,score=filterDoor(bounding_boxes,label,confidence_score)
# print(boxes,"\n\n",label,"\n\n",score)

