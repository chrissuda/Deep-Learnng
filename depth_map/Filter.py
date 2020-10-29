import numpy as np
def toNumpy(x):

    typeClass=str(type(x))
    #If x is a Torch.tensor
    if "torch" in typeClass:
        x=x.detach().cpu().numpy()

    #If x is a list
    elif "list" in typeClass:
        x=np.asarray(x)
        
    return x

def filterDoor(bounding_boxes,label,confidence_score,new_size,image_id,folder):
    """
    @param list  Object candidate bounding boxes 
    @param label predicted category
    @param list  Confidence score of bounding boxes
    @return list bounding_boxes,label,confidence_score
    """
    
    z_threshold=0.25

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

    x_right=bounding_boxes[door_index[0],2]
    y_bottom=bounding_boxes[door_index[0],3]



    #Make a threshold here and take out some doors to verify
    z0=getZ(x_right,y_bottom,new_size,image_id,folder)
    z=getZ(bounding_boxes[door_index[i],2],bounding_boxes[door_index[i],3],new_size,image_id,folder)
    door_filter_index=np.where(z-z0>z_threshold)
    door_filter_index=door_filter_index[0]

    x1=bounding_boxes[door_index[door_filter_index],0]
    y1=bounding_boxes[door_index[door_filter_index],3]
    

    x2,y2=bounding_boxes[door_index[door_filter_index],2],y_bottom

    sr_index=np.where(np.logical_or(label==3,label==4))
    sr_x1=bounding_boxes[sr_index,0]
    sr_y1=bounding_boxes[sr_index,1]

    #x1-(x2-x1)*1/2<sr_x1<sr_x1<x1+(x2-x1)*1/2
    boolean_x=np.ones((x1.size,sr_x1.size))
    sr_x1=boolean_x*sr_x1
    sr_x1=sr_x1.T
    boolean_x=np.logical_and(x1-(x2-x1)*1/2<sr_x1,sr_x1<x1+(x2-x1)*1/2)


    #y1-(y2-y1)*1/2<sr_y1<y1+(y2-y1)*1/2
    z1=getZ(x1,y1,new_size,image_id,folder)
    sr_z1=getZ(sr_x1,sr_y1,new_size,image_id,folder)
    boolean_z=np.ones((z1.size,sr_z1.size))
    sr_z1=boolean_z*sr_z1
    sr_z1=sr_z1.T

    boolean_z=np.logical_and(z1-z_threshold<sr_z1,sr_y1<z1+z_threshold)
    
    #boolean and between (boolean of x and y)
    boolean_door=np.logical_and(boolean_x,boolean_z)

    #boolean or in each row
    boolean_door=np.sum(boolean_door,axis=1)
    door_filter_index=door_filter_index[np.where(boolean_door==0)[0]]

    #Delete these labels
    bounding_boxes=np.delete(bounding_boxes,door_index[door_filter_index],axis=0)
    label=np.delete(label,door_index[door_filter_index])
    confidence_score=np.delete(confidence_score,door_index[door_filter_index])

    return bounding_boxes,label,confidence_score


# bounding_boxes=[[50,50,70,70],[10,40,30,60],[10,53,25,70],[80,40,100,60],[85,62,100,80]]
# label=[1,1,4,1,3]
# confidence_score=[0.8,0.4,0.4,0.3,0.3]
# boxes,label,score=filterDoor(bounding_boxes,label,confidence_score)
# print(boxes,"\n\n",label,"\n\n",score)

def getZ(x,y,new_size,image_id,folder):
    #Mapping from cropped_image to panorama
    pano_x,pano_y=16384,8192
    pointCloud_x,pointCloud_y=512,256
    img_width,img_height=3584.0,2560.0
    width_left_offset=2048
    width_right_offset=10752
    height_offset=3072

    y=int((y/new_size[1]*img_height+height_offset)*pointCloud_y/pano_y)
    if image_id[-5]=="0":
        x=int((x/new_size[0]*img_width+width_left_offset)*pointCloud_x/pano_x)

    elif image_id[-5]=="1":
        x=int((x/new_size[0]*img_width+width_right_offset)*pointCloud_x/pano_x)

    path=os.path.join(folder,image_id[:-6]+".npy")
    pointCloud=np.load(path)

    return pointCloud[y*img_width+x,2]

