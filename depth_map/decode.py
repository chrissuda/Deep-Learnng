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

encode_data="eJzt2n1wFPUdx_E8cZccF3IJGDDylGAsYB4ICqK3Z_YClNIKUmulZQZmqkU61Y6dOrSKhG0ZH7C02goCigrDg6LT8lDr2JhLthRKH4Kg0I4FKQXCQ6EgHUAULdDkCHeXvb3d393t_b778Hn_mewk9_u9PhxPyb8lKysnKzs_CyGEEEIIIYQQQgghhBBCCCGEEEIIIYSQXjnJRP1ikSElZY412CWD2DEDy5UheazA_HGgxwjMGVd6jMBUEdlbbwI9lFG_IAOixe-K-hI0c0WK87f6Hqjdo1HfhFouZbr-lhoCNbky6vuIKU4-SX4LrIBaWy3qOwmnbp-Sv2lXQA2dIOpr0bBPw990C6BmThjprWjbp-VvqgVQI2tEdyn6-On5m2cB1MZakV0KE396_uYYALWwdmTXwsPfBAug9tWL7GL4-FMPgJpXN7Kb4eRPOgBqXP3IroaN3wB_wgFQ4zJEdjf8_MkG4KbGZYjqbri9_dMNwA1_jXj6kwzA7bbCAChuJhxXf4IBuOGvGV9_7gNww187zv6cB-CGv068_bkOwA1_vbj7cxyAG_662djfDX_d2PgN9ec1ADf89aPw5zQA-DNkX383_Bki8ecxALfbSgPgcCHq0fhzGAD8mbKrvxv-TBH5Z3oAbjv6F2XgouBvipgOZSf_zA5AyW8L_yL4s2ZH_6IU_V3an6Xyz-QA4vgd7O9ynn88vw38i1Lzd5nWP3MDsKW_NyV_lwP9Vfid6u8ys3-mBmBP_1Te_11O9Ffjd6a_y-T-mRkA_K_G4M_IbyF_VX57-Cd5E1ftWJ4h4Yd_JP1zdfh7k7qICB7TQyT-mRiAur_ZB6B_rmT9o3hsT9nEPwE__PWegj9l-gfr9E9iADF4jI9R-Bs-gET8TvNn5Ie_SdI_WdifeQBW8Td6APAPF2vH_KDJ_X0sD8G_s2527E_Cnyz9k13xZxuAhfzZB-DTPsmV4J-lRE3iUfhTpX8y-OscJZxF_RmO3-XPMACFXTLPwp8m_YN57erPPACf3lk6g38caVIPw58k_dNH_PUGEGeX3NPm9tcdAPwt5886APiz-MfbJfm4tf0t-v9_-qeP-msPwOb-Kb8BUAtrp396Rn8VO80vC39TpH96G_uzDcDnYziOff2Lov4aA1Cz0_yypvBnGgD87evv8ehfAPz1_Yst66-_gKv-OgOAv0X99RYAf69XZwDF1vbXXgD87e-vtYCIv_aBEvmbewA6-FmZ8mfk5-efeAHw9-oMoNgW_okGAH-H-CdYQNRf80TO9S-2jb_qAuDv1R5AAn_Nr2pWf5UFwF_b32Uv_7gBxPhrnQn-NvFXLgD-Xs0B2M-_-wB8bL8BONXfZUP_2AX4fGxvAPC3k390APDX_v3fpv6RBST07xXJyf79bOvvUfOPOZUz_Qc4x__KAuDf3b_MQf4e-Ov497O3vyfO_zr4O8nf44G_wr_bABzsnwX_Dv4r_mp08LeDv1fz_R_-Tvd3WdqfYQDwhz-LvzV_ADg9fxf84Z_Kj3_B3ySl5T8E_vB3mr_KPwDBH_7whz_87e-v9h8A8Ie_Xf2L4A9_-MNfxX8Y_C3ur8tP7J9pfvgn7z806j_Emf4VNY7y9yr8IwNwhL9H6V9RkZy_mQeQvH8Z_OEPf7v739DV8AT-w0fX1dXddJOD_aurI_7V1SOrbx5pRf9rY6uqqqwcpCzev2MUw4dXfuFqKf34r4X9c3NzazsaMWJExP_GzkZZ0f-azkrDXauewr-yq5Kqqqr-V1P176-spLNrurKgf3lsgwfGZVn_-EqjxfqXRCstiUtBl-BLJ8VvIv_Bigb2VWZFfwWhilSXv2IVpSWlqsXQqXy2ZyQL-SuU87qK40-8AGpkjeJ_FSvr9Ff9hBprLK7yoyqPmNs_T7NY9-ujWcy_MFpif7UPF2ry6mZuf2153azpn7Ai1Y_a0j9N-AQToEbWiMVfvXT0c3NN5h_3Cg3dADWyRvz9c8OZxT_1DSexAWpkjbj650YzgX_q9En4h6NG1oiXf64iUv805ZP2N_EEOPgr6an9DdFP6U8I1NgqZdZflT6pCRjo3qdbFP7m20Dv3r0z468pn8QKjIDPjtanj2ErSNHfNCPo3T2j_Jnh2VZgFHz3jJhBOv7EIwjfQW_VUvdPDV5vBoayGzqDtP1JRqByC-pD0F2CgewaSzCanGkHTEMwxp_jChguIsEU1MaQnSH4RMVq53eWHjnzFBKOwUj_zM4gzSuJ3UHMhzn75-YrSvNUaZcBf6OHkNHzw59LZmOPBH-uFYSjBFcEfwL_aNTHhz_84Q9_-MMf_vDnPgDq88Mf_vCHP_zhD3_4wx_-8OfnTz0AWn_iw2fDH_7whz_84Q9_-MMf_vCHP_zhD3_4wx_-8Ic__OEPf_jDH_7wN4wf_vAnDv7whz_84Q9_-MMf_vDnOgDi88Mf_vCHP_zhD38Cf-IBkPrTHj0c_OEPf_jDH_7whz_84Q9_-MMf_vCHP_wzxw9_-FMHf_jDH_7whz_NAGjPD3_4w5_Sn3YAlP6kB-8K_vCHP_zhD3_4wx_-8Ic_F374w588-MOfeACk54c__OFP6k86AEJ_ymNHgj_84Q9_-MMf_jz54Q9_-ij8zfQXQPjT-1MOgM6f8NAxwR_-8Ic__Dnzwx_-JojE30QDIPOnO3K34A9_-MMf_nz5ne1vngFQ-ZMdWBH84Q9_7vxO9zfNAIj8qY4bF_zhD3_e_PA3ywBo_IkOqxL8Cfjhb5oBkPjTHFU1On9zDMDh_PDnzw9_8wzA6fyk_iYYgNP5af3pB-B0fmJ_8gXw5ud8PIacPQC--ibkJ_enXQBXfp4HY45aPxzZBByOn20S_1yqCTgcP9s8_p3x34Cz7TujNo-L6wgcTR-Omjthyh1kYgjOZY9G7cxYRuZgkLgRL4UsaljjipuI3kZYcLkY0EbtRhX1vZsuahA-Ud-ymcpStGD6kop-Oe7QtouS_HxFmTj6iFT_q9lH_edy2oVjb-YFL7znEfKH5oQWfy7JY-r6i4-c_YG_bdya5ncuSfK3vtRP3PPhXDl_8YBWl39vqPqzQcFtRyT5_Fsb_MefbA8VHOofHHRQqp_8_t2hN3a3C9Om9Aj-s3396oWPDQgs3CnJP7mtTGxsmrj6wN9eEi6eleRCf5k4bJEk_-srkwPvlh0JTdo_Jbhk4Nv-XeP2Nl3o-H4Vs_uKs45K9ZXr7_NvHtYuDNtwXjxX3FPY7TvbtKrj9T_zaF_xrtG_b33o9u_Ve574UWj8V59q-HTODPmenhdbV1ZuCj2wfbt4uun-lvlLpPrjv_5H85anKxo--Lsk_6GwSfhh7eFQ4ddygjv_vMC_d9gb_p91fL-RP-4pjp9RIFSWiy3PviXJ01beLM6TX_dPKOrVsukDSZ49rUC8pcc5Yc5Tua07tjTKD288KW4ceFp4b43g39EuyZUrfOJD2b8QFhUMDO0-I8knb_SK7l2SXHVhu7Bv_OHQ2ppT4r3lnwoNdT1aT_22US6pHRL89h6pfvx9a0L7-h4WKqcMDv715Dx558O_CXzuPRpqe25ZsPDwJeHje1a1nCuW5BOfHRHXnSwIvL5HkjcEtvqn7G9ueDW_uPnJYz8Xcs53fL5mhljyv_cF6fKc0LZDkrzz5BNiRc5y4e6FPxUOn5DkTZOmiw-0lDX3W345MPNyo3xo8nXBT1xrBc8Zqb7tm4uaZy6oHfvcUUnuua7UXz6rPbTxlargsdwK_--efjmwIiDJ8-98QRx57JdCy0eX3nnsY0n-ulwtfpIzIlC3Z1rL1pmSfODOXsEdz-4STs3Mbx7V8f3a8mrE_cuk-iEDJrR8sfCIsLPgO8GJY0fIr750q-zxNYY2P-Mbu-TekYEXx32j-ZHNkrzMO0o8uGq7MLTwQvO645L85fsLxE135AXm93rNf26fJG__j0f84-ryQPaWtcKTsiQ_uOOCOOFgjZyz_7b6WYubQre-eKZhy6lpwsuPLg2d-UiSc95dXj9v0Gjh9NSloa3_7XD4y_L68nFnhQUHTzR_96Ak3zVsqvhaozcw8tj0Md_vuNMZbY-L6yceFa6vHNpa--BcuebNtuDSO_q2vJBXe_vEXnPrJzW8HfQ9vqNp7iv_DhyYOk_-09QbgiueHxBY-eGY0OQ2SZ4ytFH8P0tORx4"
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

def visualizeDepth(depthMap):
    im = depthMap["depthMap"]
    im[np.where(im == max(im))[0]] = 255
    if min(im) < 0:
        im[np.where(im < 0)[0]] = 0
    im = im.reshape((depthMap["height"], depthMap["width"])).astype(int)
    # display image
    plt.imshow(im)
    plt.show()


def visualize(xyz):
    xyz=xyz.reshape(-1,3)
    maxx=np.amax(xyz,axis=0)
    minn=np.amin(xyz,axis=0)
    xyz=(xyz-minn)/(maxx-minn)
    print(xyz.shape)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud("sync.ply", pcd)

    # Load saved point cloud and visualize it
    pcd_load = o3d.io.read_point_cloud("sync.ply")

    # Convert Open3D.o3d.geometry.PointCloud to numpy array
    xyz_load = np.asarray(pcd_load.points)
    print('xyz_load')
    print(xyz_load.shape)
    o3d.visualization.draw_geometries([pcd_load])

	# o3d.visualization.draw_geometries([pcd], zoom=0.3412,
 #                                  front=[0.4257, -0.2125, -0.8795],
 #                                  lookat=[2.6172, 2.0475, 1.532],
 #                                  up=[-0.0694, -0.9768, 0.2024])

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
# pointCloud = depthMap["pointCloud"]
# print("depthMap and pointCloud created")
# print("pointcloud size:",pointCloud.size)
# print("depth map size:",depthMapData.size)
# vector=[]
# for i in range(0,int(pointCloud.size),3):
# 	vector.append([pointCloud[i],pointCloud[i+1],pointCloud[i+2]])
#vector=np.array(vector)
# np.save("pointCloud.npy",pointCloud)
#vector=np.load("pointCloud.npy")
# 
# vector=vector.reshape((-1,3))
# print(vector.shape)
visualizeDepth(depthMap)
# print(pcData(256,234,vector))
# print(pcData(120,234,vector))

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


