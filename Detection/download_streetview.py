import urllib.request
import json
import os
'''
params={
	"size":640x640,
	"location":"lat,lng", or a string representation of a location
	"heading":, the angle of an image. N:0,360  E:90  S:180
	"fov": the zoom level of an image. Default is 90. The number
	should decrease if you zoom in
	"source": default; outdoor 
	"key": Your API key
}
'''

params=[]
size="640x640"
key="AIzaSyDblzztJ7voz0ZTddkX_hEeYXrMoup0GY8"
source="outdoor"
fov=100
heading=210

def download_Streetview(params,img_file):

	folder,file_name=os.path.split(img_file)
	string=''
	url_meta="https://maps.googleapis.com/maps/api/streetview/metadata?"
	url_img="https://maps.googleapis.com/maps/api/streetview?"


	#Unpack the dict and assemble an url
	for k,v in params.items():
		string+=k+"="+str(v)+"&"

	string=str(string.rstrip("&"))
	url_meta=url_meta+string
	url_img=url_img+string
	
	#Download image file
	print(url_img)
	urllib.request.urlretrieve(url_img,img_file)

	#Download json file
	request = urllib.request.urlopen(url_meta)
	data = json.load(request)
	data.pop("copyright")
	data["name"]=file_name
	with open(os.path.join(folder,"meta.json"), 'a+') as f:
		data=json.dump(data,f,indent=4,separators=(',', ': '))
		f.write(",\n")

def download():
	for i in range(1,2):
		dictonary={}
		url="https://maps.googleapis.com/maps/api/geocode/json?address="+str(i*5)+"W+33rd+St+New+York+NY&key="+key
		r = urllib.request.urlopen(url)
		location = json.load(r)
		lat=location["results"][0]["geometry"]["location"]["lat"]
		lng=location["results"][0]["geometry"]["location"]["lng"]

		dictonary["size"]=size
		dictonary["location"]=str(lat)+","+str(lng)	
		dictonary["source"]=source;
		dictonary["fov"]=fov;
		dictonary["heading"]=heading
		dictonary["key"]=key

		download_Streetview(dictonary,"google.jpg")

