'''
This file provides several scripts to access the geographic location of where the image was taken, 
as well as the 3-d information for a specific point in the image
'''

import numpy as np
from util_detection import *
import torch
import csv

def getZ(x,y,pointCloud,new_size,image_id):
    """
    get the z-value (height) of a specific point in an image

    Args:
        x (nd.array): an array of x-value of points in an image

        y (nd.array): an array of y-value of points in an image

        pointCloud (nd.array): the point cloud information of an image. Its shape is[256,512,3], 
                                while the first and second dimension of the array represents the coordinates of a point along with height and width dimension of an image,
                                the third dimension represent the x,y,z point-cloud value for that pixel

        new_size (2-tuple): the sizes of an image after resizing before feeding into the model, 
                            in (width,height)

        image_id (str): an unique identifer of an image in the dataset.

    Returns:
        nd.array: an array of z-values of points in an image
    """

    #map the x,y coordinates of an image to its point cloud    
    x,y=mapCoord(x,y,new_size,image_id)

    #Return z-axis value
    return pointCloud[y,x,2]

def mapCoord(x,y,new_size,image_id):
    """
    map the x,y coordinates of a point in the image into the coordinates of its point cloud

    Args:
        x (nd.array): an array of x-value of points in an image

        y (nd.array): an array of y-value of points in an image

        new_size (2-tuple): the sizes of an image after resizing before feeding into the model, 
                            in (width,height)

        image_id (str): an unique identifer of an image in the dataset.

        folder (str): a path to the folder where your xml files and images are stored.

    Raises:
        AttributeError: if the image_id is invalid

    Returns:
        nd.array: an array of latitude of points in an image
        nd.array: an array of longitude of points in an image
    """

    #Convert to numpy format
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

    return x,y

def getLatLon(x,y,new_size,image_id,folder):
    """
    get the latitude and longitude of multiple points in an image

    Args:
        x (nd.array): an array of x-value of points in an image

        y (nd.array): an array of y-value of points in an image

        new_size (2-tuple): the sizes of an image after resizing before feeding into the model, 
                            in (width,height)

        image_id (str): an unique identifer of an image in the dataset.

        folder (str): a path to the folder where your xml files and images are stored.

    Raises:
        AttributeError: if the image_id is invalid

    Returns:
        nd.array: an array of latitude of points in an image. 0 if this point is not in valid range
        nd.array: an array of longitude of points in an image. 0 if this point is not in valid range
    """

    #set the max_distance relative to the origin that a valid point could not exceeed 
    max_distance=999
    
    #Read the geologic information of this image
    clat,clon,yaw = getGeoOfImage(os.path.join(folder,image_id[:-6] + ".xml"))

    #Read the point cloud information of this image
    pointCloud=np.load(os.path.join(folder,image_id[:-6]+".npy"))

    #Convert yaw degree
    if(yaw > 180):
        yaw = yaw - 180
    else:
        yaw = 180 + yaw

    #map the x,y coordinates of an image to its point cloud    
    x,y=mapCoord(x,y,new_size,image_id)

    #Calculate lat lon
    dx = pointCloud[y,x,0]
    dy = pointCloud[y,x,1]

    #get the index of points that are not valid if their distance exceed maximum distance
    index_null=np.logical_or(dx>=max_distance,dy>=max_distance)[0]

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


def getGeoOfImage(xml_path):
    """
    find the latitude and longitude and yaw angle of where this image was taken

    Args:
        path_to_metadata_xml (str): the path to the .xml file associated with this image

    Returns:
        float: latitude of where the image was taken
        float: longitude of where the image was taken
        yaw: the compass heading of the camera relative with Truth North. 0~360 in clockwise direction
    """
    pano = {}
    pano_xml = open(xml_path, 'rb')
    tree = ET.parse(pano_xml)
    root = tree.getroot()
    for child in root:
        if child.tag == 'projection_properties':
            pano[child.tag] = child.attrib
        if child.tag == 'data_properties':
            pano[child.tag] = child.attrib
    
    return float(pano['data_properties']['lat']),float(pano['data_properties']['lng']), float(pano['projection_properties']['pano_yaw_deg'])




def saveLatLonPrediction(model,device,loader,newSize,folder,csv_file):
    """
    save the lat-lon of predictions output by model into a csv file

    Args:
        model (pytorch Faster RCNN): 

        device (): torch.device("cuda") or torch.device("cpu")

        loader (): a generation which loads the dataset.
                    batch size must be 1

        new_size (2-tuple): the sizes of an image after resizing before feeding into the model, 
                            in (width,height) 
        folder (str): a path to the folder where your xml files and images are stored.

        csv_file (str): a path to the .csv file to save lat-lon predictions 

    Raises:
        ValueError: if the batch size of loder is not 1
    """
    if loader.batch_size!=1:
        raise ValueError("loader must have a batch size of 1")

    with open(csv_file,'w') as f:
        writer=csv.writer(f)

    for i,(x, y) in enumerate(loader):
        # move to device, e.g. GPU
        x=x.to(device=device, dtype=torch.float32)
        #Get Predictions
        target = model(x)

        #Filter using nms and snms
        target[0]["boxes"],target[0]["labels"],target[0]["scores"]=nms(target[0]["boxes"],target[0]["labels"],target[0]["scores"],0.4)
        target[0]["boxes"],target[0]["labels"],target[0]["scores"]=snms(target[0]["boxes"],target[0]["labels"],target[0]["scores"],0.1)


        box=toNumpy(target[0]["boxes"])

        x=(box[:,0]+box[:,2])/2
        y=(box[:,1]+box[:,3])/2
        lat,lon=getLatLon(x,y,folder,newSize,y[0]["image_id"])
        
        for i in range(lat.size):
            writer.writerow([y[0]["image_id"],target[0]["labels"][i],target[0]["scores"][i],lat[i],lon[i]])


# def getPlane(x,y,plane,new_size,image_id):
    
#     #Mapping from cropped_image to panorama
#     pano_x,pano_y=16384,8192
#     pointCloud_x,pointCloud_y=512,256
#     img_width,img_height=3584,2560
#     width_left_offset=2048
#     width_right_offset=10752
#     height_offset=3072

#     y=(y/new_size[1]*img_height+height_offset)*pointCloud_y/pano_y
#     if image_id[-5]=="0":
#         x=(x/new_size[0]*img_width+width_left_offset)*pointCloud_x/pano_x

#     elif image_id[-5]=="1":
#         x=(x/new_size[0]*img_width+width_right_offset)*pointCloud_x/pano_x


#     if type(x)==int:
#         x=int(x)
#         y=int(y)
#     else:
#         x=x.astype(int)
#         y=y.astype(int)

    
#     #Return plane indics
#     return plane[y,x]
