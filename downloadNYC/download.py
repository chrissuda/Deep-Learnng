import csv
import urllib.request
import json
import os
import io
from PIL import Image
import random

try:
    from xml.etree import cElementTree as ET
except ImportError as e:
    from xml.etree import ElementTree as ET

'''
apiKey1="https://maps.googleapis.com/maps/api/streetview?size=400x600&location=40.7561812,-73.9812787
&key=AIzaSyDblzztJ7voz0ZTddkX_hEeYXrMoup0GY8"

apiKey2="https://maps.googleapis.com/maps/api/streetview?size=400x600&pano=MMxVBkGROmpb9ECs3CIPqg
&key=AIzaSyDblzztJ7voz0ZTddkX_hEeYXrMoup0GY8"
'''

def readJson():
    
    with open('road.json') as json_file:
        data = json.load(json_file)
    
    return data


#Get width and height of GSV image given xml data of image
def getWidthHeight(path_to_metadata_xml):
    pano = {}
    pano_xml = open(path_to_metadata_xml, 'rb')
    tree = ET.parse(pano_xml)
    root = tree.getroot()
    #find and return width and height as array
    for child in root:
        if child.tag == 'data_properties':
            pano[child.tag] = child.attrib
    return (int(pano['data_properties']['image_width']),int(pano['data_properties']['image_height']))


def getPanoId(latlon):
    #API call with latlon => returns xml file with data of image including pano_id
    url = "http://maps.google.com/cbk?output=xml&ll=" + str(latlon[0]) + "," + str(latlon[1]) + "&dm=1"
    try:
        xml = urllib.request.urlopen(url)
        tree = ET.parse(xml)
        root = tree.getroot()
        pano = {}
        #Find and returns pano_id
        for child in root:
            if child.tag == 'data_properties':
                pano[child.tag] = child.attrib
        return pano['data_properties']['pano_id']

    #Handle the exception if the request is not successful
    except Exception as e:
        print(e)
        return None


def downloadImg(pano_id,folder):
	
    base_url = 'http://maps.google.com/cbk?'
    url_param = 'output=tile&zoom=' + str(5) + '&x=' + str(x) + '&y=' + str(
                y) + '&cb_client=maps_sv&fover=2&onerr=3&renderer=spherical&v=4&panoid=' + pano_id
    url = base_url + url_param

    response = requests.get(url, stream=True)
    if response.ok:
        file = open(folder+pano_id+".jpg", "wb")
        file.write(response.content)
        file.close()
    else:
        print(response.reason)
    

def readLatLon(file):
    '''
    @param file a csv_file you want ot read, which contains lat and lon
    '''
    data=[] #[{pano_id,lat,lon,street}]
    pano_ids=[]
    success=0
    skip=0
    fail=0

    #Get the number of total data
    with open(file,'r') as f:
        total = len(list(csv.DictReader(f)))
    f.close()

    with open(file,'r') as f:
        reader = csv.DictReader(f)
        for line in reader:
                
            lat=float(line["lat"])
            lon=float(line["lon"])
            
            latlon=(lat,lon)
            pano_id = getPanoId(latlon)

            if pano_id!=None:
                #If the pano_id is already existed in our database
                if pano_id in pano_ids:
                    skip+=1

                #Successfully get the pano_id
                else:
                    line["pano_id"]=pano_id
                    data.append({"pano_id":pano_id,"lat":lat,"lon":lon})
                    pano_ids.append(pano_id)
                    success+=1

            #Request is failed
            else:
                fail+=1
            

            print("Total:",total," Success:",success," Fail:",fail," Skip:",skip)

    return data

def savePanoId(file,data):
    '''
    @param file a csv file you want to save to
    @param data:[{pano_id,lat,lon,street},]
    '''
    
    with open(file,'w') as f:
        fieldnames =data[0].keys() #header
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(data)

def downloadSingleXML(output_folder,pano_id):
    #download xml data for image using pano_id to query
    url = "http://maps.google.com/cbk?output=xml&cb_client=maps_sv&hl=en&dm=1&pm=1&ph=1&renderer=cubic,spherical&v=4&panoid="+pano_id
    xml = urllib.request.urlopen(url)

    outputFile=os.path.join(output_folder,pano_id)

    #Return 0 status code if it has been downloaded.
    if os.path.exists(outputFile+".xml"):
        return 0

    with open(outputFile+".xml",'wb') as f:
        for line in xml:
            f.write(line)

    return xml.getcode()


def downloadSinglePano(output_folder,pano_id):
    #file: _0 left side of the panorama
    #      _1 right side of the panorama

    outputFile=os.path.join(output_folder,pano_id)

    #Terminate the function if xml file not found
    if not os.path.exists(outputFile+".xml"):
        return "XML not found"

    #get width and height of this image 
    wh = getWidthHeight(outputFile+".xml")
    image_width = wh[0]
    image_height = wh[1]
    im_dimension = (image_width, image_height)

    if im_dimension!=(16384,8192):
        return "panorama dimension is not 16384*8192"
        
    #Using 7 tiles in x-axis, and 5 tiles in y-axis. Assume 32*16 tiles in total
    x_tiles,y_tiles=7,5
    im_crop_dimension=(int(im_dimension[0]/32*x_tiles),int(im_dimension[1]/16*y_tiles))
    
    '''
    Calculate which tile to use
    Assume the first tile is at 1 instead of 0;Assume there are 32*16 tiles
    range:  x_tile_left:[5,12) (2048,3072)->(5632,5632)
            x_tile_right:[22,29) (10752,3072)->(14336,5632)
            y_tile_range:[7,12)
        
    '''
    x_tile_left_range=(int(int(round(image_width / 512.0))/8+1),int(int(round(image_width / 512.0))/8+1+x_tiles))
    x_tile_right_range=(int(int(round(image_width / 512.0))/2+6),int(int(round(image_width / 512.0))/2+6+x_tiles))
    x_tile_ranges=[x_tile_left_range,x_tile_right_range]
    y_tile_range=(int(int(round(image_height/ 512.0))/2-1),int(int(round(image_height/ 512.0))/2)-1+y_tiles)
    
    base_url = 'http://maps.google.com/cbk?'

    #loop through two sides of a panoramas
    for i in range(len(x_tile_ranges)):
        #blank canvas to paste tiles
        blank_image = Image.new('RGB', im_crop_dimension, (0, 0, 0, 0))
        #Loop through as many tiles as in the image given each tile is 512
        for y in range(y_tile_range[0]-1,y_tile_range[1]):
            for x in range(x_tile_ranges[i][0]-1,x_tile_ranges[i][1]):
                #API call with at specific tile and zoom
                #http://maps.google.com/cbk?output=tile&zoom=5&x=&y=&cb_client=maps_sv&fover=2&onerr=3&renderer=spherical&v=4&panoid=MMxVBkGROmpb9ECs3CIPqg
                url_param = 'output=tile&zoom=' + str(5) + '&x=' + str(x) + '&y=' + str(
                    y) + '&cb_client=maps_sv&fover=2&onerr=3&renderer=spherical&v=4&panoid=' + pano_id
                url = base_url + url_param

                # Open an image, resize it to 512x512, and paste it into a canvas
                req = urllib.request.urlopen(url)
                file = io.BytesIO(req.read())

                im = Image.open(file)
                im = im.resize((512, 512))
                
                blank_image.paste(im, (512 * (x-x_tile_ranges[i][0]+1), 512 * (y-y_tile_range[0]+1)))

                #Save the image
            blank_image.save(outputFile+"_"+str(i)+".jpg")
            #I changed this 436 from 664 for permission
            os.chmod(outputFile+"_"+str(i)+".jpg", 436)
        

    return req.getcode()

def downloadPano(downloaded_csv,csv_file,output_folder):

    pano_ids=[]
    downloaded_pano_ids_csv=[]
    data=[]

    #Collect pano id which have already been downloaded
    with open(downloaded_csv, "r") as f:
        for line in csv.DictReader(f):
            downloaded_pano_ids_csv.append(str(line["pano_id"]))
            
            
    #Collcet pano id that needs to be downloaded
    with open(csv_file, "r") as f:
        for line in csv.DictReader(f):
            pano_id=str(line["pano_id"])

            if pano_id not in downloaded_pano_ids_csv:
                pano_ids.append(pano_id)
                data.append(line)
                

    #Rewrite pano id into csv
    with open(csv_file,'w') as f:
        fieldnames =data[0].keys() #header
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(data)


    pano_ids
    total=len(pano_ids)
    success=0
    skip=0 #If it has already been downloaded
    failure=0 #If it failed to be downloaded

    for p in pano_ids:
        print("\npanoid:",p)
        xmlCode=downloadSingleXML(output_folder,p)
        #See if it has been downloaded before
        if xmlCode==0:
            skip+=1
            print("Skip")
            continue

        panoCode=downloadSinglePano(output_folder,p)

        if xmlCode==200 and panoCode==200:
            success+=1

        else:
            status={"Http XML status":xmlCode,"Http PANO status":panoCode}
            print("*Download Failure",status," pano_id:",p)
            failure+=1

            #Remove file if it fails to download
            
            try:
                os.remove(os.path.join(output_folder,str(p)+".xml"))
            except:
                pass
            try:
                os.remove(os.path.join(output_folder,str(p)+"_0.jpg"))
            except:
                pass
            try:
                os.remove(os.path.join(output_folder,str(p)+"_1.jpg"))
            except:
                pass

        print("Total:",total," Success:",success," Failure:",failure," Skip:",skip)


def compareCSVandPano(csv_file,pano_folder):
    '''
    See which downloaded pano is not recorded in csv_file
    @return a list of missing pano_id 
    '''

    pano_ids_csv=[]
    
    pano_ids_xml=[]
    missing_pano_ids=[]

    

    #Collect pano id that are needed to download
    with open(csv_file, "r") as f:
        for line in csv.DictReader(f):
            pano_id=str(line["pano_id"])

            pano_ids_csv.append(pano_id)
            

    for filename in os.listdir(pano_folder):
        if filename.endswith(".xml"):
            pano_ids_xml.append(filename[0:-4])


    for pano_id_xml in pano_ids_xml:
        if pano_id_xml not in pano_ids_csv:
            missing_pano_ids.append(pano_id_xml)

            print("Missing csv PANO id:",pano_id_xml)

    print("Total Missing  Pano:",len(missing_pano_ids))

    return missing_pano_ids

downloadSinglePano("./","MMxVBkGROmpb9ECs3CIPqg")

