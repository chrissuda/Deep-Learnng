# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 22:18:33 2020
@author: Jerry
"""

import urllib
from urllib import request
import os
import io
from PIL import Image

try:
    from xml.etree import cElementTree as ET
except ImportError as e:
    from xml.etree import ElementTree as ET

import base64
import zlib
import numpy as np
import struct
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from functools import cmp_to_key
import csv
import math
#import geopy.distance
import sys

def getLonLat():
    return [40.716012, -74.000290]

def getPanoId(lonLat):
    url = "http://maps.google.com/cbk?output=xml&ll=" + str(lonLat[0]) + "," + str(lonLat[1]) + "&dm=1"
    xml = urllib.request.urlopen(url)
    tree = ET.parse(xml)
    root = tree.getroot()
    pano = {}
    for child in root:
        if child.tag == 'data_properties':
            pano[child.tag] = child.attrib
    return pano['data_properties']['pano_id']

def getDepthMap(path_to_metadata_xml):
    pano_xml = open(path_to_metadata_xml, 'rb')
    tree = ET.parse(pano_xml)
    root = tree.getroot()
    for child in root:
        if child.tag == 'model':
            root = child[0]
    return root.text;
    
def getWidthHeight(path_to_metadata_xml):
    pano = {}
    pano_xml = open(path_to_metadata_xml, 'rb')
    tree = ET.parse(pano_xml)
    root = tree.getroot()
    for child in root:
        if child.tag == 'data_properties':
            pano[child.tag] = child.attrib
    return (int(pano['data_properties']['image_width']),int(pano['data_properties']['image_height']))

def downloadPano():
    url = "http://maps.google.com/cbk?output=xml&cb_client=maps_sv&hl=en&dm=1&pm=1&ph=1&renderer=cubic,spherical&v=4&panoid="
    xml = urllib.request.urlopen(url + pano_id)
    with open(output_file + ".xml", 'wb') as f:
        for line in xml:
            f.write(line)
    wh = getWidthHeight(output_file + ".xml")
    image_width = wh[0]
    image_height = wh[1]
    im_dimension = (image_width, image_height)
    blank_image = Image.new('RGB', im_dimension, (0, 0, 0, 0))
    
    base_url = 'http://maps.google.com/cbk?'
    
    for y in range(int(round(image_height / 512.0))):
        for x in range(int(round(image_width / 512.0))):
            url_param = 'output=tile&zoom=' + str(5) + '&x=' + str(x) + '&y=' + str(
                y) + '&cb_client=maps_sv&fover=2&onerr=3&renderer=spherical&v=4&panoid=' + pano_id
            url = base_url + url_param

            # Open an image, resize it to 512x512, and paste it into a canvas
            req = urllib.request.urlopen(url)
            file = io.BytesIO(req.read())

            im = Image.open(file)
            im = im.resize((512, 512))

            blank_image.paste(im, (512 * x, 512 * y))
    
    blank_image.save(output_file + '.jpeg')
    #change to 664
    os.chmod(output_file + '.jpeg', 664)

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

def computeDepthMap(header, indices, planes):
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
                #depthMap[y * w + (w - x - 1)] = t
                depthMap[y*w + (w-x-1)] = t
                pointCloud[3 * y * w + 3 * x] = v[0] * t 
                pointCloud[3 * y * w + 3 * x + 1] = v[1] * t
                pointCloud[3 * y * w + 3 * x + 2] = v[2] * t
            else:
                #depthMap[y * w + (w - x - 1)] = 9999999999999999999.0
                depthMap[y*w + (w-x-1)] = 9999999999999999999.0
                pointCloud[3 * y * w + 3 * x] = 9999999999999999999.0
                pointCloud[3 * y * w + 3 * x + 1] = 9999999999999999999.0
                pointCloud[3 * y * w + 3 * x + 2] = 9999999999999999999.0
    return {"width": w, "height": h, "depthMap": depthMap, "pointCloud": pointCloud}

def pcData(x, y, pointCloud):
    return str(x) + " " + str(y) + ": " + str(pointCloud[3*(y * 512 + x)]) + " " + str(pointCloud[3*(y * 512 + x) + 1]) + " " + str(pointCloud[3*(y * 512 + x) + 2])

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
'''
def findLatLon(panoid):
    #Google API Key
    key = "AIzaSyAptRS0n1nzV9LpJhD2f8p3V7gKvRGZFjE"
    url = "https://maps.googleapis.com/maps/api/streetview/metadata?pano=" + panoid + "&key=" + key
    jsonData = requests.get(url)
    data = jsonData.json()
    return (data["location"]["lat"], data["location"]["lng"])
'''
def latLonMap(pointCloud):
    latLon = np.empty(512*256*2)
    for x in range(512):
        for y in range(256):
            dx = pointCloud[(512*y + x) * 3]
            dy = pointCloud[(512*y + x) * 3 + 1]      
            if(dx == 9999999999999999999.0 or dy == 9999999999999999999.0):
                continue
            rdx = dx*np.cos(np.radians(yaw)) + dy*np.sin(np.radians(yaw))
            rdy = -1*dx*np.sin(np.radians(yaw)) + dy*np.cos(np.radians(yaw))
            dlat = rdy / 111111
            dlon = rdx / (111111 * np.cos(np.radians(clat)))
            latLon[2*(y*512 + x)] = dlat + clat
            latLon[2*(y*512 + x) + 1] = dlon + clon
    return latLon

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
        if(pointCloud[(512*y + x) * 3] != 9999999999999999999.0):
            rdx = pointCloud[(512*y + x) * 3]*np.cos(np.radians(yaw)) + pointCloud[(512*y + x) * 3 + 1]*np.sin(np.radians(yaw))
            rdy = -1*pointCloud[(512*y + x) * 3]*np.sin(np.radians(yaw)) + pointCloud[(512*y + x) * 3 + 1]*np.cos(np.radians(yaw))
            currD = np.sqrt(rdx**2+rdy**2)
            if(currD > dist):
                inPict = True
            if(np.abs(dist - currD) < minD):
                minD = np.abs(dist - currD)
                miny = y
    if not inPict:
        return[9999999999999999999.0, 9999999999999999999.0]
    return [x, miny]
    
def treeCmp(a, b):
    if a[0] > b[0]:
        return 1
    if a[0] < b[0]:
        return -1
    return 0

def getTreeData(filename):
    treeLL = []
    with open(filename, 'r') as csvfile: 
        # creating a csv reader object 
        csvreader = csv.reader(csvfile, quotechar = '"') 
      
        # extracting field names through first row 
        next(csvreader) 
  
        # extracting each data row one by one 
        for row in csvreader: 
            if row[6] == "Alive":
                treeLL.append((float(row[37]), float(row[38])))
    treeLL = sorted(treeLL, key=cmp_to_key(treeCmp))
    return treeLL

def writeSortedTreeData(treeLL):
    with open('sortedTreeLL.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for x in treeLL:
            writer.writerow([x[0], x[1]])

def getTreeDataFromSorted(filename):
    treeLL = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile) 
        for row in csvreader:
            treeLL.append((float(row[0]), float(row[1])))
    return treeLL

def deg2rad(degrees):
    return math.pi*degrees/180.0

def rad2deg(radians):
    return 180.0*radians/math.pi

# Semi-axes of WGS-84 geoidal reference
WGS84_a = 6378137.0  # Major semiaxis [m]
WGS84_b = 6356752.3  # Minor semiaxis [m]

# Earth radius at a given latitude, according to the WGS-84 ellipsoid [m]
def WGS84EarthRadius(lat):
    # http://en.wikipedia.org/wiki/Earth_radius
    An = WGS84_a*WGS84_a * math.cos(lat)
    Bn = WGS84_b*WGS84_b * math.sin(lat)
    Ad = WGS84_a * math.cos(lat)
    Bd = WGS84_b * math.sin(lat)
    return math.sqrt( (An*An + Bn*Bn)/(Ad*Ad + Bd*Bd) )

# Bounding box surrounding the point at given coordinates,
# assuming local approximation of Earth surface as a sphere
# of radius given by WGS84
def boundingBox(latitudeInDegrees, longitudeInDegrees, halfSideInKm):
    lat = deg2rad(latitudeInDegrees)
    lon = deg2rad(longitudeInDegrees)
    halfSide = 1000*halfSideInKm

    # Radius of Earth at given latitude
    radius = WGS84EarthRadius(lat)
    # Radius of the parallel at given latitude
    pradius = radius*math.cos(lat)

    latMin = lat - halfSide/radius
    latMax = lat + halfSide/radius
    lonMin = lon - halfSide/pradius
    lonMax = lon + halfSide/pradius

    return (rad2deg(latMin), rad2deg(lonMin), rad2deg(latMax), rad2deg(lonMax))

def latBSearch(treeLL, target):
    l = 0
    r = len(treeLL) - 1
    while l < r:
        m = (int) ((l+r) / 2)
        if treeLL[m][0] < target:
            l = m + 1
        else:
            r = m - 1
    return l

def findTreesInBox(treeLL, latRange, minL, maxL):
    result = []
    for i in range(latRange[0], latRange[1] + 1):
        if treeLL[i][1] >= minL and treeLL[i][1] <= maxL:
            result.append((treeLL[i][0], treeLL[i][1]))
    return result
    
def drawImage():
    data = plt.imread(output_file + ".jpeg")
    fig, ax = plt.subplots(figsize=(32, 16), dpi=96)
    ax.imshow(data, interpolation='none')
    plt.axis('off')
    for ar in treeCoords:
        ic = findImageCoord(ar[0], ar[1], yaw, clat, clon, pointCloud)
        if ic != [9999999999999999999.0, 9999999999999999999.0]:
            #print(geopy.distance.distance(ar, (latLonMap[2*(ic[1]*512 + ic[0])], latLonMap[2*(ic[1]*512 + ic[0]) + 1])).km*1000)
            square = patches.Rectangle((ic[0] * 32, ic[1] * 32), 50, 50, color='RED')
            ax.add_patch(square)
    
    plt.show()
    saveImagePath =pano_id + ".jpeg"
    plt.savefig(saveImagePath, bbox_inches='tight', pad_inches=0)

lonLat = getLonLat()
pano_id = getPanoId([lonLat[0], lonLat[1]])

output_file =pano_id

#downloadPano()   
print("downloaded image: " + pano_id)
depthMap = getDepthMap(output_file + ".xml")
# decode string + decompress zip
depthMapData = parse(depthMap)
# parse first bytes to describe data
header = parseHeader(depthMapData)
# parse bytes into planes of float values
data = parsePlanes(header, depthMapData)

# compute position and values of pixels
depthMap = computeDepthMap(header, data["indices"], data["planes"])
pointCloud = depthMap["pointCloud"]
print("depthMap and pointCloud created")

np.savetxt('test.txt', pointCloud) 

latlon = findLatLon(output_file + ".xml")
clat = latlon[0]
clon = latlon[1]
yaw = latlon[2]
print(clat, clon)
if(yaw > 180):
    yaw = yaw - 180
else:
    yaw = 180 + yaw

latLonMap = latLonMap(pointCloud)
print("latLonMap created")
bounds = boundingBox(clat, clon, 0.04)

'''
treeData = '2015_Street_Tree_Census_-_Tree_Data.csv'
treeLL = getTreeData(treeData)
writeSortedTreeData(treeLL)
'''

# treeLL = getTreeDataFromSorted('sortedTreeLL.csv')

# latRange = [latBSearch(treeLL, bounds[0]), latBSearch(treeLL, bounds[2])]
# treeCoords = findTreesInBox(treeLL, latRange, bounds[1], bounds[3])
# print("found tree coords")
#drawImage()

def findTreeCoords():
    for ar in treeCoords:
        ic = findImageCoord(ar[0], ar[1], yaw, clat, clon, pointCloud)
        if ic != [9999999999999999999.0, 9999999999999999999.0]:
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

def drawFacade(facade):
    data = plt.imread(output_file + ".jpeg")
    fig, ax = plt.subplots(figsize=(32, 16), dpi=96)
    ax.imshow(data, interpolation='none')
    plt.axis('off')
    for ar in facade:
        square = patches.Rectangle((ar[0] * 32, ar[1] * 32), 50, 50, color='RED')
        ax.add_patch(square)
    
    plt.show()
    saveImagePath =pano_id + ".jpeg"
    plt.savefig(saveImagePath, bbox_inches='tight', pad_inches=0)

#drawFacade(findFacade(latLonMap))

def facadePlanes(indices):
    result = []
    for x in range(512):
        arr = []
        for y in range(255, -1, -1):
            if indices[y * 512 + x] not in arr:
                result.append([x,y])
                arr.append(indices[y * 512 + x])
    return result

edgePlanes = []

def floodfill(indices, x, y, prev, visited):
    if x < 0 or y < 0 or x >= 512 or y >= 256 or (y * 512 + x) in visited:
        return
    visited.add(y * 512 + x)
    if indices[y * 512 + x] != prev:
        edgePlanes.append([x, y])
    floodfill(indices, x + 1, y, indices[y * 512 + x], visited)
    floodfill(indices, x - 1, y, indices[y * 512 + x], visited)
    floodfill(indices, x, y + 1, indices[y * 512 + x], visited)
    floodfill(indices, x, y - 1, indices[y * 512 + x], visited)
                
#drawFacade(facadePlanes(data["indices"]))
#sys.setrecursionlimit(15000)

floodfill(data["indices"], 0, 0, 0, set())
drawFacade(edgePlanes)