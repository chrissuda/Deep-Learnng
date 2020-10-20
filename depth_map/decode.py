import urllib
import os
import io
from PIL import Image
import open3d as o3d
try:
    from xml.etree import cElementTree as ET
except ImportError as e:
    from xml.etree import ElementTree as ET

import base64
import zlib
import numpy as np
import struct
#import requests
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from functools import cmp_to_key
import csv
import math
#import geopy.distance

encode_data="eJzt3X1wFHcdx_EkhAskF47A8RSaJ57SpAQhFFrIbm4PZMSpRco4LVTKlCm0nUqpJaDCAKu1BYrVlsbUIlYtlBTRAYOKE-4uP0FBLTpNixULDII49oFWYCjClELd3CXhcrd3ew-7-_3d7uf9FznIzP1-r88dj0P6TMnKysnK7pOFEEIIIYQQQgghhBBCCCGEEEIIIYQQQkirnDhRPzekf_G8taJ-7ijl0mHHEjI5neUxg8zJSHqsgOtMoscIOMx0e2yAn8jwsQH6qPFDUd-CTaNmD4_6LuwXtXhk1Pdhr6i1VaO-FNtEDR0z6ouxRdTIcaO-HMtHDawZ9QVZOmrchKK-JOtGLZto1PdkzahVk4j6qqwYtWlyUd-W5aIGTTbq-4qqd_yon55G1JzJR31joTTUM2UJ1JipRHxlKcjzOgNqyVQjuzCHw5EuP0cjoGZMPYrbcoTSx5-HDVAjppPZd-XoTj9_4g1QE6aXmTflCE9ff8INUAummVnX5IhIf3-aCVD7pZ0ptxSpb5B_Xl6eKce5UQE1X_oZfkfR-Ab6mzuBAvhrpapvqL95Eyiwgr-hA4ihb7S_ORMosIa_cQOIqW-CvwkLsIq_QQOIo2-Kv9ETKIB_3DjwN3IBBdbxN2YAXPgbtwAr-RsyANP51f0NWkAB_DXixt-QBVjL34gBcORvwALgrxVX_roPwGL-BgyAL3-9F2A1f_0HwJu_vgOAv1bc-eu7AKv56z4ADv31HAD8NeLRX8cFWM5f7wHw6a_bAOCvEaf-eg0A_hrx6q_TAuCvEb_-ugwA_hpx7K_HAOCvEc_-OgwA_hpx7Z_-AOAfr_z8Mr790x4A_NXdu-LdP90BWO4vANL1z-8Z9_5pDgD-MekzxD-9AVjOX0f7DPF3wD8sPfEzxD-dAcBf-S1epvunMQD4K7_Dz3j_1AcAf0v4pzwA-Dss4Z_qAGzv77Ckf7w5uODfkz-efzx-rvwd_eDfkX39leCfiH_Y5Tis5d-9gDi_HnDB38L-nROAP_zhH4PfFv6xBuCCP_zt7O-AP_x18TeGXy9_9QG44A9_O_v3vBn429rfYWV_tQG44A9_S_vfpM0Pf_jbwV9lAPC3h3-MNwCX3f0jXhfwhz_87ePvsJd_1ADgD3_4wx_-8Ic__G3mP9ze_i74wx_-UX_9C3_428DfZW__EfCHP_zhD3_4w98u_t3_IAz-8Ic__G3lH3ZWF_zhD3-b-Y-phH-wm0dkDYe_PfyHwD9Rf8t9AYCgvxL84Q9_-MMf_hr-cfjhn2np60_w37_CP63gD3_4w7_rfF3fgD_84Q9_pXHB4G8v_6qqqs6bqgr2GSVL-pd3FHqs3DGmvFRp0iTlIZerPORfHpbl_J1OZ7XS2GBDu1K8Ox6tDt5KSWcTJ1rSf7xSaeix0ohC_mEP1NZazT8oO0GppqPbO6lrqqtvCVVyw7-kxJL-xR05HMUxcrl6fGg1_8LCwqE9uvVWBVqBH6qSJf1HRje5f8ws6K-SYq32sCX9Y1tHNLAji_lXqNZhrXYDlvQfEN3A2FnkH4A44xbvVWAVfxX2RPw7F0Dtl07x7TX8h4XKbP949lr-o5Uy2H9UZLH8h2mVgf65Yd2WoP_o8LK7o3ZMKfVf7EXnVHi13yVU4tc_N6myE4qaM6kStQ_6pxqXX_41Ofok_DNmAsnYpzWAePwU_inQJ-fP_QaSpreIf6ryqQyAzxEMSo0-Zf_BSjz4F4Vljr87FDV4eIO6M95_8I16EfqHLIpUMs7fHRE1e7DsQSoZ4B_u3iNT_dVh0p5BsvLBQp9KSd-jNHaQkHoEe2QG-yf0Ak15B0nBR386sXxEyQ8hsRd6Yunsn5B7QkOIuwRtd80N0cMnuIXoNTidqUhrZbq6eqpbiFyDCnpCbxxmjUDPC4nOCP3k5tDH2ANqlhK28Ssw6fSm-HfVRzWTThorff1T3wHN6eFvjL_mEPp2RXt6-BvsH15ftWhPD3_4wx_-8Ic__OEPf_jDH_4mDoD2AuAPf_jDH_7whz-FP_EAiP1pDw9_-MMf_vCHP_zhD3_4wx_-8DeJH_7wpwz-8Ic__OEPf_jDH_7whz_84Q9_-MMf_vCHP_zhD3_4wx_-8Ic__PUcAO0NwB_-8Kf0px0ArT_p0bPhD3_4wx_-8Ic__OFvMj_84U8a_OEPf_jDH_7whz_84Q9_-MMf_vCHvykDIL0B-JP7kw6A1J_y4MHgD3_4wx_-8Ie_2fzwhz9t8Ic__OEPf_jDH_40A6C8AfjT-1MOgNKf8NidwR_-8Cfihz_8iYM__OEPf_jD354DIPSnO3R38Ic__Gn44Q9_6uAPf3sPgM6f7MhhwR_-8Cfhhz_8ySPw52wAZP5UB-4R_OEPfwp--PMxACp_ouNGROHP1xsAET_84c9BJP5cDYDIn-awUcGfht_e_jwNwNb8VP4cDYDEn-KgqsGfgh_-_AzA1vx0_twMwNb8hP68DMDW_JT-nPw5sNn8Jh9PI0p_Pt4CzNXnjJ_Yn4cF2Fmf3p_-ZwHz9E07UhJR6wcjnYCN8bM58e-IagM2tu-Imj0i80dgX_pg1OAxMu_dwJ7s3VFDJ1SMNegyB53E9XgqFFHT6lOsgWhvJBFeUyCIopajjvr-qaO-f-qo75-6Xr2yIvrdsW2vLJh30Tfhusz8I4ulR56WPVcONojHtzmF4mVOr9S4S2g9vKe15ILMcpXvP374t_57r5X5vnhOZo1zS6SlNWvZu0suBn5add23aLXLe3qzzHa0LwwEflzov2eg0zvxbzL71eSiwNKrRf79C4d4by7b-srasy-JpwSZFRQWS833fH_Ukzn-wPSza9mW8mFS5cxp7IFfj2EbG1qnNmRt97IPZc-ih_cIj37sFraPz_WeO1Minv_A6Rv1uszanxku5Rz7vdCw7Ae-xg9ktuJ6hfR640J29PKF-jXvLvF_b12zd8MPZc_jAxaKw39WKKzo39-7c5tYN2fPx1O3fCqzPznc0s4DOYG_3ldS33JkNVswRJYmtTb7Nq4cKfz7qsyEGWOlptyNU9uPVAsPfSKzqhVjpaO_OOm_s-lFsWmCzM7lzJIcr-aIbR-Wt53YvZot37RGal_0bWH-tzaLtR6ZuQa-6Zkx-EFP89xTbXunPCzM9bNpB4vXC0953tiXfVlm0_cMkI5OXiyuXze_rV_TStZ671vS9RW7fXvLKuveVjxONNRId--Ume-OWYHnd_Xzv3Mu3_uPc-M8Hx2fyq6dekJ4YfaL0y5UXRZ-dHKvb_RpmZ3eMFaaUXnEN-yRu6ZsUz6_91fGS481ba6_VLOced5bJYy7elnqX7HbP3PlT8RrVcrzm7_Qu_O95roN_Z7wbVV-_KelFdLX7v8Gu_jR19tOSwd8eYPyp7kLjtUVv3nI7zkrs2f_2OzJahHFrzbuD_yvcS1beXGDVDQ9x3_oSrn4hcMy61c6R6rctIzNe3pb_eduu9vf8puHph1ukdlj7Z8N3FHr8pdcr_Y2N61h75Q523bsu-S7dMTv3Vjr2HfwfMuUa4rHU71ypPNDi8T63f_15_lk9sKWTVJe4IzQ65_Zvu_8R2YDDt8kzTtxQXjyLwvqCs_IbF_f9z3uFrd4cfbbvvuOyGzrrFppUu1zgjtQ3jbzz6tZu_-Kd8jJZ4VvNg8KjHxNZgum7pWEHRX-0hPrxceXyOyX23O90_Pd4jP_ejDwvvLxz4tbpRNfrhHE_bX-pVdk9tLnT3n-cChHXOx-LlCu7HXdJ72963feIs6580tt1ZNWsRHsLu_i10YIbxzIrXtZ2ccB0SPl9X_Zt2X2Hv8u5fVROn-vxz3273WPNp8MVK5ay_Ju_2798_dfEJY_8Kpw_C3l-bAVnv8Da7Z0CQ"
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
    saveImagePath = "C:/Allan/Streetview/dataCollection/" + pano_id + ".jpeg"
    plt.savefig(saveImagePath, bbox_inches='tight', pad_inches=0)



def visualize():
	pcd = o3d.io.read_point_cloud("pointCloud.txt",format="xyz")
	print(pcd)
	print(np.asarray(pcd.points))
	o3d.visualization.draw_geometries([pcd], zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

# lonLat = getLonLat()
# pano_id = getPanoId([lonLat[0], lonLat[1]])

# output_file = "C:/Allan/Streetview/panoramas/" + pano_id

# #downloadPano()   
# print("downloaded image: " + pano_id)
# depthMap = getDepthMap(output_file + ".xml")
# decode string + decompress zip
depthMapData = parse(encode_data)
# parse first bytes to describe data
header = parseHeader(depthMapData)
# parse bytes into planes of float values
data = parsePlanes(header, depthMapData)

#compute position and values of pixels
depthMap = computeDepthMap(header, data["indices"], data["planes"])
pointCloud = depthMap["pointCloud"]
# print("depthMap and pointCloud created")
# print("pointcloud size:",pointCloud.size)
# print("depth map size:",depthMapData.size)
# vector=[]
# for i in range(0,int(pointCloud.size),3):
# 	vector.append([pointCloud[i],pointCloud[i+1],pointCloud[i+2]])
#vector=np.array(vector)
vector=np.load("pointCloud.npy")
# np.save("pointCloud.npy",vector)
print(vector.shape)
# print(pcData(256,180,vector))
# print(pcData(511,180,vector))

# print(pcData(128,128,vector))
# print(pcData(384,128,vector))

# print(vector)
	# if int(pointCloud[i])==0 and int(pointCloud[i+1]==0):
	# 	print("z:",pointCloud[i:i+3]," i:",i)
	# elif int(pointCloud[i+1])==0 and int(pointCloud[i+2]==0):
	# 	print("x:",pointCloud[i:i+3]," i:",i)
	# elif int(pointCloud[i])==0 and int(pointCloud[i+2]==0):
	# 	print("y:",pointCloud[i:i+3]," i:",i)

#np.save("pointCloud.npy",vector)
# postiveY=pointCloud[(83961-1)*3:(83961-1)*3+3]
# print("+y:",postiveY)
# negativeY=pointCloud[(74489-1)*3:(74489-1)*3+3]
# print("-y:",negativeY)
# postiveX=pointCloud[(56963-1)*3:(56963-1)*3+3]
# print("+x:",postiveX)
# negativeX=pointCloud[(15214-1)*3:(15214-1)*3+3]
# print("-x:",negativeX)

# latlon = findLatLon(output_file + ".xml")
# clat = latlon[0]
# clon = latlon[1]
# yaw = latlon[2]
# print(clat, clon)
# if(yaw > 180):
#     yaw = yaw - 180
# else:
#     yaw = 180 + yaw

# latLonMap = latLonMap(pointCloud)
# print("latLonMap created")
# bounds = boundingBox(clat, clon, 0.04)

# '''
# treeData = '2015_Street_Tree_Census_-_Tree_Data.csv'
# treeLL = getTreeData(treeData)
# writeSortedTreeData(treeLL)
# '''

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
    saveImagePath = "C:/Allan/Streetview/facade/" + pano_id + ".jpeg"
    plt.savefig(saveImagePath, bbox_inches='tight', pad_inches=0)


