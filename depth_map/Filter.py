import numpy as np
import os
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
    print("bounding_boxes:",bounding_boxes)
    print("scores:",confidence_score)
    print("label:",label)
    
    pointCloud=np.load(os.path.join(folder,image_id[:-6]+".npy"))

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
    z0=getZ(x_right,y_bottom,pointCloud,new_size,image_id)
    print("x_right:",x_right,"y_bottom:",y_bottom)
    print("z0:",z0)
    z=getZ(bounding_boxes[door_index,2],bounding_boxes[door_index,3],pointCloud,new_size,image_id)
    print("z:",z)
    door_filter_index=np.where(z-z0>z_threshold)
    door_filter_index=door_filter_index[0]
    print("door_filter_index:",door_filter_index)

    x1=bounding_boxes[door_index[door_filter_index],0]
    y1=bounding_boxes[door_index[door_filter_index],3]
    

    x2,y2=bounding_boxes[door_index[door_filter_index],2],y_bottom

    sr_index=np.where(np.logical_or(label==3,label==4))
   
    sr_x1=bounding_boxes[sr_index,0]
    sr_y1=bounding_boxes[sr_index,1]

    #x1-(x2-x1)*1/2<sr_x1<sr_x1<x1+(x2-x1)*1/2
    boolean_x=np.ones((x1.size,sr_x1.size))
    sr_x=boolean_x*sr_x1
    sr_x=sr_x.T
    boolean_x=np.logical_and(x1-(x2-x1)*1/2<sr_x,sr_x<x1+(x2-x1)*1/2)


    #z+threshold<sr_z1<z1+z_threshold
    z1=getZ(x1,y1,pointCloud,new_size,image_id)
    sr_z=getZ(sr_x1,sr_y1,pointCloud,new_size,image_id)
    boolean_z=np.ones((z1.size,sr_z.size))
    sr_z=boolean_z*sr_z
    sr_z=sr_z.T

    boolean_z=np.logical_and(z1-z_threshold<sr_z,sr_z<z1+z_threshold)
    
    #boolean and between (boolean of x and y)
    boolean_door=np.logical_and(boolean_x,boolean_z)

    #boolean or in each column
    #print(door_filter_index,boolean_door)
    boolean_door=np.sum(boolean_door,axis=0)

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
    
    
    index=y*pointCloud_x+x
    if type(index)==int:
        index=int(index)
    else:
        index=index.astype(int)

    #Return z-axis value
    return pointCloud[index,2]


def getlatlon(x,y,pointCloud,clat,clon,yaw):
    x=int(x)
    y=int(y)

    print("x",x,"y:",y)

    if(yaw > 180):
        yaw = yaw - 180
    else:
        yaw = 180 + yaw

    

    dx = pointCloud[y,x,0]
    dy = pointCloud[y,x,1]    

    print("dx",dx,"dy:",dy)

    if(dx >= 500 or dy >= 500):
        return
    rdx = dx*np.cos(np.radians(yaw)) + dy*np.sin(np.radians(yaw))
    rdy = -1*dx*np.sin(np.radians(yaw)) + dy*np.cos(np.radians(yaw))
    print("rdx",rdx,"rdy:",rdy)

    dlat = rdy / 111111
    dlon = rdx / (111111 * np.cos(np.radians(clat)))
    print("dlat",dlat,"dlon:",dlon)
    
    lat = dlat + clat
    lon = dlon + clon

    print(lat,lon)

#[814.908447265625, 414.46807861328125, 921.9517822265625, 616.8416748046875]
x=900
y=500
pano_x,pano_y=16384,8192
pointCloud_x,pointCloud_y=512,256
img_width,img_height=3584,2560
width_left_offset=2048
width_right_offset=10752
height_offset=3072
new_size=(1000,1000)

y=(y/new_size[1]*img_height+height_offset)*pointCloud_y/pano_y
# if image_id[-5]=="0":
x=(x/new_size[0]*img_width+width_left_offset)*pointCloud_x/pano_x

# elif image_id[-5]=="1":
#     x=(x/new_size[0]*img_width+width_right_offset)*pointCloud_x/pano_x


pointCloud=np.load("TAhKmMfAex974GGcmXr_8g.npy")

clat = 40.761848
clon = -73.975335
yaw = 298.18
getlatlon(x,y,pointCloud,clat,clon,yaw)