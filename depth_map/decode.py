import urllib
import os
import io
from PIL import Image
from tqdm import tqdm
import open3d as o3d
try:
    from xml.etree import cElementTree as ET
except ImportError as e:
    from xml.etree import ElementTree as ET

import collections
import base64
import zlib
import numpy as np
import struct
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from functools import cmp_to_key
import csv
import math

#Define the max_distance that a point can not exceed
MAX_DISTANCE=999

def getDepthMap(path_to_metadata_xml):
    pano_xml = open(path_to_metadata_xml, 'rb')
    tree = ET.parse(pano_xml)
    root = tree.getroot()
    for child in root:
        if child.tag == 'model':
            root = child[0]
    return root.text;

def parse(b64_string):
    # fix the 'inccorrect padding' error. The length of the string needs to be divisible by 4.
    b64_string += "=" * ((4 - len(b64_string) % 4) % 4)
    # convert the URL safe format to regular format.
    data = b64_string.replace("-", "+").replace("_", "/")

    data = base64.b64decode(data)  # decode the string
    data = zlib.decompress(data)  # decompress the data
    return np.array([d for d in data])

def parseHeader(depthMap):
    return {
        "headerSize": depthMap[0],
        "numberOfPlanes": getUInt16(depthMap, 1),
        "width": getUInt16(depthMap, 3),
        "height": getUInt16(depthMap, 5),
        "offset": getUInt16(depthMap, 7),
    }

def get_bin(a):
    ba = bin(a)[2:]
    return "0" * (8 - len(ba)) + ba

def getUInt16(arr, ind):
    a = arr[ind]
    b = arr[ind + 1]
    return int(get_bin(b) + get_bin(a), 2)

def getFloat32(arr, ind):
    return bin_to_float("".join(get_bin(i) for i in arr[ind : ind + 4][::-1]))

def bin_to_float(binary):
    return struct.unpack("!f", struct.pack("!I", int(binary, 2)))[0]

def parsePlanes(header, depthMap):
    indices = []
    planes = []
    n = [0, 0, 0]

    for i in range(header["width"] * header["height"]):
        indices.append(depthMap[header["offset"] + i])

    for i in range(header["numberOfPlanes"]):
        byteOffset = header["offset"] + header["width"] * header["height"] + i * 4 * 4
        n = [0, 0, 0]
        n[0] = getFloat32(depthMap, byteOffset)
        n[1] = getFloat32(depthMap, byteOffset + 4)
        n[2] = getFloat32(depthMap, byteOffset + 8)
        d = getFloat32(depthMap, byteOffset + 12)
        planes.append({"n": n, "d": d})

    return {"planes": planes, "indices": indices}

def computeDepthMap(header, indices, planes,img_file):

    v = [0, 0, 0]
    w = header["width"]
    h = header["height"]

    depthMap = np.empty(w * h)
    pointCloud = np.empty(w * h * 3)

    sin_theta = np.empty(h)
    cos_theta = np.empty(h)
    sin_phi = np.empty(w)
    cos_phi = np.empty(w)

    for y in range(h):
        theta = (h - y - 1) / (h - 1) * np.pi
        sin_theta[y] = np.sin(theta)
        cos_theta[y] = np.cos(theta)

    for x in range(w):
        phi = (w - x - 1) / (w - 1) * 2 * np.pi + np.pi / 2
        sin_phi[x] = np.sin(phi)
        cos_phi[x] = np.cos(phi)

    for y in range(h):
        for x in range(w):
            planeIdx = indices[y * w + x]

            v[0] = sin_theta[y] * cos_phi[x]
            v[1] = sin_theta[y] * sin_phi[x]
            v[2] = cos_theta[y]

            if planeIdx > 0:
                plane = planes[planeIdx]
                t = np.abs(
                    plane["d"]
                    / (
                        v[0] * plane["n"][0]
                        + v[1] * plane["n"][1]
                        + v[2] * plane["n"][2]
                    )
                )
                depthMap[y*w + (w-x-1)] = t
                pointCloud[3 * y * w + 3 * x] = v[0] * t
                pointCloud[3 * y * w + 3 * x + 1] = v[1] * t 
                pointCloud[3 * y * w + 3 * x + 2] = v[2] * t 
            else:
                depthMap[y*w + (w-x-1)] = 0
                pointCloud[3 * y * w + 3 * x] =MAX_DISTANCE 
                pointCloud[3 * y * w + 3 * x + 1] =MAX_DISTANCE
                pointCloud[3 * y * w + 3 * x + 2] = MAX_DISTANCE

    pointCloud_width=512
    pointCloud_height=256
    pointCloud=pointCloud.reshape(pointCloud_width,pointCloud_height,3)

    # Open the image form working directory and Resize it
    if img_file:
        image = Image.open(img_file)
        size=(w,h)
        image=image.resize(size)
        tensor=np.asarray(image)
        tensor=tensor.copy()
        tensor=tensor.astype("float")
        tensor/=255.0
        tensor=tensor.reshape(w*h,3)
        
        pointCloud=np.concatenate((pointCloud,tensor),axis=1)

    return {"width": w, "height": h, "depthMap": depthMap, "pointCloud": pointCloud}




def pcData(x, y, pointCloud):
    return str(x) + " " + str(y) + ": " + str(pointCloud[3*(y * 512 + x)]) + " " + str(pointCloud[3*(y * 512 + x) + 1]) + " " + str(pointCloud[3*(y * 512 + x) + 2])

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


def getLatLonOfImage(panoid):
    """
    find the latitude and longitude  of where this image was taken 
    by requesting google street view api

    Args:
        panoid (str): the unique identifier of the panorama

    Returns:
        float: latitude of where the image was taken
        float: longitude of where the image was taken
    """

    #Google API Key
    key = "AIzaSyAptRS0n1nzV9LpJhD2f8p3V7gKvRGZFjE"
    url = "https://maps.googleapis.com/maps/api/streetview/metadata?pano=" + panoid + "&key=" + key
    jsonData = requests.get(url)
    data = jsonData.json()
    return (data["location"]["lat"], data["location"]["lng"])


def findImageCoord(lat, lon, yaw, clat, clon, pointCloud):
    dy = 111111 * (lat - clat)
    dx = 111111 * np.cos(np.radians(clat)) * (lon - clon)
    
    dangle = np.degrees(np.arctan(dy/dx))
    #check if dx and dy is 0??
    if dx < 0 and dy > 0:
        dangle += 180
    elif dx < 0 and dy < 0:
        dangle += 180
    elif dx > 0 and dy < 0:
        dangle += 360
    if dangle < 90:
        dangle = 90 - dangle
    else:
        dangle = 450 - dangle
    offset = dangle - yaw
    if offset < 0:
        offset += 360
    
    x = (int)((offset / 360) * 512)
    
    dist = np.sqrt(dx**2+dy**2)
    minD = 99999999999999999
    miny = -1
    inPict = False
    for y in range(255, -1, -1):
        if(pointCloud[(512*y + x) * 3] != MAX_DISTANCE):
            rdx = pointCloud[(512*y + x) * 3]*np.cos(np.radians(yaw)) + pointCloud[(512*y + x) * 3 + 1]*np.sin(np.radians(yaw))
            rdy = -1*pointCloud[(512*y + x) * 3]*np.sin(np.radians(yaw)) + pointCloud[(512*y + x) * 3 + 1]*np.cos(np.radians(yaw))
            currD = np.sqrt(rdx**2+rdy**2)
            if(currD > dist):
                inPict = True
            if(np.abs(dist - currD) < minD):
                minD = np.abs(dist - currD)
                miny = y
    if not inPict:
        return[MAX_DISTANCE, MAX_DISTANCE]
    return [x, miny]
    


def deg2rad(degrees):
    return math.pi*degrees/180.0

def rad2deg(radians):
    return 180.0*radians/math.pi




def WGS84EarthRadius(lat):
    '''
    Earth radius at a given latitude, according to the WGS-84 ellipsoid [m]
    Semi-axes of WGS-84 geoidal reference
    http://en.wikipedia.org/wiki/Earth_radius
    '''
    WGS84_a = 6378137.0  # Major semiaxis [m]
    WGS84_b = 6356752.3  # Minor semiaxis [m]
    
    An = WGS84_a*WGS84_a * math.cos(lat)
    Bn = WGS84_b*WGS84_b * math.sin(lat)
    Ad = WGS84_a * math.cos(lat)
    Bd = WGS84_b * math.sin(lat)
    return math.sqrt( (An*An + Bn*Bn)/(Ad*Ad + Bd*Bd) )


def visualizeDepth(depthMap):
    im = depthMap["depthMap"]
    im[np.where(im == max(im))[0]] = 255
    if min(im) < 0:
        im[np.where(im < 0)[0]] = 0
    im = im.reshape((depthMap["height"], depthMap["width"])).astype(int)
    # display image
    plt.imshow(im)
    plt.show()


def visualize(format="xyzrgb"):
    
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(xyz)

    # Load saved point cloud and visualize it
    pcd_load = o3d.io.read_point_cloud("pointCloud.txt",format=format)
    o3d.io.write_point_cloud("pointCloud.ply", pcd_load)

    
    o3d.visualization.draw_geometries([pcd_load])

    # Example
    # generate some neat n times 3 matrix using a variant of sync function
    # x = np.linspace(-3, 3, 401)
    # mesh_x, mesh_y = np.meshgrid(x, x)
    # z = np.sinc((np.power(mesh_x, 2) + np.power(mesh_y, 2)))
    # z_norm = (z - z.min()) / (z.max() - z.min())
    # xyz = np.zeros((np.size(mesh_x), 3))
    # xyz[:, 0] = np.reshape(mesh_x, -1)
    # xyz[:, 1] = np.reshape(mesh_y, -1)
    # xyz[:, 2] = np.reshape(z_norm, -1)
    # print('xyz')
    # print(xyz)

    # # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(xyz)
    # o3d.io.write_point_cloud("sync.ply", pcd)

    # # Load saved point cloud and visualize it
    # pcd_load = o3d.io.read_point_cloud("sync.ply")
    # o3d.visualization.draw_geometries([pcd_load])



def findTreeCoords():
    for ar in treeCoords:
        ic = findImageCoord(ar[0], ar[1], yaw, clat, clon, pointCloud)
        if ic != [MAX_DISTANCE, MAX_DISTANCE]:
            print(ic, ar, end = ", ")

#findTreeCoords()

def findFacade(latLonMap):
    facade = []
    for x in range(512):
        prevLat = -1
        prevLon = -1
        for y in range(255, -1, -1):
            currLat = latLonMap[2*(y*512 + x)]
            currLon = latLonMap[2*(y*512 + x) + 1]
            if(currLat == 0 or currLon == 0):
                continue
            #print(prevLat, currLat, prevLon, currLon)
            dist = geopy.distance.distance([prevLat, prevLon], [currLat, currLon]).km*100000
            if(prevLon != -1 and prevLat != -1 and dist < 1):
                facade.append([x, y])
                break
            prevLat = currLat
            prevLon = currLon
    return facade

def drawFacade(facade,output_file,pano_id):
    data = plt.imread(output_file + ".jpeg")
    fig, ax = plt.subplots(figsize=(32, 16), dpi=96)
    ax.imshow(data, interpolation='none')
    plt.axis('off')
    for ar in facade:
        square = patches.Rectangle((ar[0] * 32, ar[1] * 32), 50, 50, color='RED')
        ax.add_patch(square)
    
    plt.show()
    saveImagePath = "C:/Allan/Streetview/facade/" + pano_id + ".jpeg"
    plt.savefig(saveImagePath, bbox_inches='tight', pad_inches=0)


# xml_file="TJ87kEIgfY3DiwT89eskfw.xml"
# img_file="TJ87kEIgfY3DiwT89eskfw.jpeg"
# decode_data = getDepthMap(xml_file)
# #decode string + decompress zip
# depthMapData = parse(decode_data)
# # # parse first bytes to describe data
# header = parseHeader(depthMapData)
# # # parse bytes into planes of float values
# data = parsePlanes(header, depthMapData)


# #compute position and values of pixels
# depthMap = computeDepthMap(header, data["indices"], data["planes"],img_file)

# if depthMap['pointCloud'].shape[1]==6:
#     format="xyzrgb"
# else:
#     format="xyz"

# # #visualize point cloud
# visualize()


def savePointCloud(folder):
    """
    decode the depth map information into point cloud inside the .xml file
    and save it in .npy format

    Args:
        folder (str): the folder path where your .xml files are saved
    """
    for f in tqdm(os.listdir(folder)):
        if f.endswith(".xml"):
            f=os.path.join(folder,f)
            depthMap = getDepthMap(f)
            # decode string + decompress zip
            depthMapData = parse(depthMap)
            # parse first bytes to describe data
            header = parseHeader(depthMapData)
            # parse bytes into planes of float values
            data = parsePlanes(header, depthMapData)
            #compute position and values of pixels
            depthMap = computeDepthMap(header, data["indices"], data["planes"])
            pointCloud = depthMap["pointCloud"]
            np.save(f[:-4]+".npy",pointCloud)

def getZ(x,y,pointCloud):
    #Mapping from cropped_image to panorama
    pano_x,pano_y=16384,8192
    pointCloud_x,pointCloud_y=512,256
    img_width,img_height=3584,2560
    width_left_offset=2048
    width_right_offset=10752
    height_offset=3072

    y=y*pointCloud_y/pano_y
    x=x*pointCloud_x/pano_x
    
    y=int(y) 
    x=int(x)


    #Return z-axis value
    return pointCloud[y,x,:2]

def savePlane(folder):
    for f in tqdm(os.listdir(folder)):
        if f.endswith(".xml"):
            f=os.path.join(folder,f)

            #Read the planes data
            depthMap = getDepthMap(f)
            depthMapData = parse(depthMap)
            header = parseHeader(depthMapData)
            data = parsePlanes(header, depthMapData)
            planes = data["indices"]
            planes=np.asarray(planes)
            planes=planes.reshape(256,512)
            np.save(f[:-4]+"_plane.npy",planes)  

# xml_file="_kZgMDYln1dUd5AcdETOkg.xml"
#savePlane("/home/students/cnn/NYC_PANO")
# npy="/home/students/cnn/Deep-Learnng/depth_map/MMxVBkGROmpb9ECs3CIPqg.npy"
# pointCloud2=np.load(npy)


# origin=getZ(8192,8191,pointCloud2)
# target=getZ(11776,4321,pointCloud2)
# print(target)

# angle=math.atan(target[0]/target[1])
# angle=math.degrees(angle)
# print(angle)

# x,y=11776,4321
# lat,lon=getLatLon(x,y,'/home/students/cnn/Deep-Learnng/depth_map/MMxVBkGROmpb9ECs3CIPqg.xml',npy)
# print(lat,lon)
# print(pointCloud2[144,424,:],"\n")


# print(pointCloud2[135,422,:],"\n")


# print(pointCloud2[145,426,:],"\n")


# print(pointCloud2[133,414,:],"\n")


# print(pointCloud2[145,402,:],"\n")


# print(pointCloud2[146,402,:],"\n")


# print(pointCloud2[137,416,:],"\n")
