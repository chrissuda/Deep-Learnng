from pickle import *
from PIL.Image import new
import torch
import torchvision
import json
from torchvision import transforms
import sys
sys.path.insert(0,"/home/students/cnn/Deep-Learnng")

from Detection.Loader import*
from Detection.CocoDataset import labelboxCoco
from Detection.util_labelbox import *
from Detection.util_detection import*
from Detection.util_train import*
from Detection.util_filter import*
from Detection.util_geolocation import *
import csv 
import shutil
import os

base_model_path="./base_model.pt"
model_path="./model.pt"
batch_size=6
train_annFile="./Data/NYCcoco.json"
val_annFile="./Data/NYCcoco.json"

img_root="/home/students/cnn/NYC"
newSize=(1000,1000)
epochs=5


#Using GPU or CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cuda.cufft_plan_cache.clear()
else:
    device = torch.device('cpu')

transform = transforms.Compose([
                transforms.Resize(newSize),
                transforms.ToTensor()])


train_labelbox=labelboxCoco(img_root,train_annFile,newSize,transform=transform)
loader_train=Loader(train_labelbox,batch_size=batch_size,shuffle=True)
# print(len(loader_train))
val_labelbox=labelboxCoco(img_root,val_annFile,newSize,transform=transform)
loader_val=Loader(val_labelbox,end=100,batch_size=batch_size)
print(len(loader_val))

# count(loader_val)
# test(loader_val)

#Set up the model
# model=torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# #4 categories + 1 background
# model=transfer(model,5)
model=torch.load(base_model_path,map_location=device)
# # print(model)
# for param in model.parameters():
#     param.requires_grad = True
# for param in model.roi_heads.parameters():
#     param.requires_grad = True
# #for name, param in model.named_parameters():
#     #print(name, param.requires_grad)
optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,betas=(0.9, 0.95))
# model=train(model,optimizer,epochs,
#                 loader_train=loader_train,loader_val=loader_val,device=device,wb=False)


# folder="../NYC"
# predictOnImageFolder(folder,model_path,0.3)
# predictOnImageFolder(folder,model_path,0.3,NMS=True)
model.eval()
# loader_val=Loader(val_labelbox,batch_size=batch_size)
# ap=getAp(model,loader_val,device,THRESHOLD_SCORE=0,NMS=True,isFilter=False)
# printAp(ap)
#1VasDr0SGipapiH85V4TJg_1
#3Gjjm-amf2-K-3y_VE_wYQ_1
#1J9kGrNztc2pQwvz-aC27g_1
# with open("latlon.csv",'w') as f:
#     writer=csv.writer(f)
    

# loader_val=Loader(val_labelbox,batch_size=1)
# for i,(x, y) in enumerate(loader_val):

#     # if y[0]["image_id"]=="EfKRbHhV_Wpty6-DAmUSUg_0.jpg":
#     # move to device, e.g. GPU
#     x=x.to(device=device, dtype=torch.float32)
#     target = model(x)

#     # print("\n\n",y[0]["image_id"])
#     target[0]["boxes"],target[0]["labels"],target[0]["scores"]=nms(target[0]["boxes"],target[0]["labels"],target[0]["scores"],0.4)
#     target[0]["boxes"],target[0]["labels"],target[0]["scores"]=snms(target[0]["boxes"],target[0]["labels"],target[0]["scores"],0.1)


#     target[0]["boxes"],target[0]["labels"],target[0]["scores"]=filterDoor(target[0]["boxes"],target[0]["labels"],target[0]["scores"]
#     ,(1000,1000),y[0]["image_id"],folder="/home/students/cnn/NYC_PANO")
#     #     target[0]["boxes"],target[0]["labels"],target[0]["scores"]=Filter(target[0]["boxes"],target[0]["labels"],target[0]["scores"])

#     box=toNumpy(target[0]["boxes"])
#     lat,lon=getLatLon(box[:,0],box[:,1],"/home/students/cnn/NYC_PANO",newSize,y[0]["image_id"])
    
    # for i in range(lat.size):
    #     writer.writerow([y[0]["image_id"],target[0]["labels"][i],target[0]["scores"][i],lat[i],lon[i]])

#     # target[0]["boxes"],target[0]["labels"],target[0]["scores"]=removeDoors(target[0]["boxes"],target[0]["labels"],target[0]["scores"],
#     # y[0]["image_id"],folder="/home/students/cnn/NYC_PANO")

    # predict_file="../Deniz/"+y[0]["image_id"][:-4]+"_predict_filter.jpg"
    # truth_file="../Deniz/"+y[0]["image_id"][:-4]+"_truth.jpg"
    # draw(x[0],target[0],file=predict_file)
#     draw(x[0],y[0],file=truth_file)
    
# except Exception as e:
# 	print("\ni:",i," image_id:",y[0]["image_id"],e,"\n")

results=turnIntoDatabase(model,loader_val,device,(1,1),newSize,"/home/students/cnn/NYC")
with open("database.json",'w') as f:
    json.dump(results,f)