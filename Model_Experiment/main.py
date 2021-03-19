from func import getMeanStd
from pickle import *
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

base_model_path="base_model.pt"
model_path="model.pt"
batch_size=6
train_annFile="../Detection/NYCCoco.json"
val_annFile="../Detection/NYCCoco.json"

img_root="/home/students/cnn/NYC"
newSize=(1000,1000)
epochs=6


#Using GPU or CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cuda.cufft_plan_cache.clear()
else:
    device = torch.device('cpu')

transform = transforms.Compose([
                transforms.Resize(newSize),
                transforms.ToTensor()])#transforms.Normalize([0.4100, 0.3981, 0.3867],[0.2228, 0.2159, 0.2184])

#"./Data/UCFcoco.json","./Data/CMPcoco.json","./Data/DOORKNOBcoco.json"]
ucf=labelboxCoco("/home/students/cnn/all",["./Data/UCFcoco.json"],newSize,transform=transform)
# loader_train=Loader(ucf,start=100,batch_size=batch_size,shuffle=True)
# loader_ucf=Loader(ucf,end=100,batch_size=batch_size,shuffle=True)
# #print(len(loader_train))

# loader=Loader(ucf,start=200,end=300,batch_size=batch_size)
# loader_ucf=Loader(ucf,end=100,batch_size=batch_size)

nyc=labelboxCoco("/home/students/cnn/all",["./Data/NYCcoco.json"],newSize,transform=transform)
loader_train=Loader(nyc,start=200,batch_size=batch_size,shuffle=True)
loader=Loader(nyc,start=600,end=700,batch_size=batch_size,shuffle=True)
loader_nyc=Loader(nyc,end=100,batch_size=batch_size)
#loader_nyc_copy=Loader(nyc,start=200,end=400,batch_size=batch_size)
# loader_val=Loader(labelbox,end=100,batch_size=batch_size)
# print(len(loader_val))



# count(loader_val)
# test(loader_val)

#Set up the model
#model=torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
#4 categories + 1 background
#model=transfer(model,5)
#model=torch.load("/home/students/cnn/Deep-Learnng/Model_Experiment/models/model.pt",map_location=device)
# print(model)

#print(model)
# for param in model.parameters():
#     param.requires_grad = False
# for param in model.rpn.parameters():
#     param.requires_grad = True
# for param in model.roi_heads.parameters():
#     param.requires_grad = True


# for name, param in model.named_parameters():
#     print(name, param.requires_grad)
#optimizer=torch.optim.Adam(model.parameters(),lr=8e-5,betas=(0.9, 0.95))
#model=train(model,optimizer,epochs,
#             loader_train=loader_train,loader_val=[loader,loader_nyc],device=device,wb=False)

#torch.save(model,"/home/students/cnn/Deep-Learnng/Model_Experiment/models/model.pt")



# model.eval()
# ap=getAp(model,loader_nyc,device,NMS=True)
# printAp(ap)

model=torch.load(model_path)
model.eval()
# ap=getAp(model,loader_nyc,device,NMS=True)
# printAp(ap)



data=[]
loader=Loader(nyc,end=200,batch_size=1)
for i,(x, y) in enumerate(loader):

    # if y[0]["image_id"]=="EfKRbHhV_Wpty6-DAmUSUg_0.jpg":
    # move to device, e.g. GPU
    x=x.to(device=device, dtype=torch.float32)
    target = model(x)
    
    

    # # print("\n\n",y[0]["image_id"])
    # target[0]["boxes"],target[0]["labels"],target[0]["scores"]=nms(target[0]["boxes"],target[0]["labels"],target[0]["scores"],0.4)
    # target[0]["boxes"],target[0]["labels"],target[0]["scores"]=snms(target[0]["boxes"],target[0]["labels"],target[0]["scores"],0.1)


#     target[0]["boxes"],target[0]["labels"],target[0]["scores"]=filterDoor(target[0]["boxes"],target[0]["labels"],target[0]["scores"]
#     ,(1000,1000),y[0]["image_id"],folder="/home/students/cnn/NYC_PANO")
#     #     target[0]["boxes"],target[0]["labels"],target[0]["scores"]=Filter(target[0]["boxes"],target[0]["labels"],target[0]["scores"])


    # predict_file=os.path.join("Model_Experiment","experiment",y[0]["image_id"][:-4]+"_predict.jpg")
    # truth_file=os.path.join("Model_Experiment","experiment",y[0]["image_id"][:-4]+"_truth.jpg")
    # draw(x[0],target[0],file=predict_file)
    # draw(x[0],y[0],file=truth_file)

#Open File
dataFile=open("./Data/output.json",'w')
json.dump(data,dataFile)


def getDoor(y,target):
    #Take out door only
    labels=target[0]["labels"]
    door=np.where(labels==1)[0]
    door_boxes=target[0]["boxes"][door]
    door_scores=target[0]["scores"][door]
    
    doorTruth=np.where(y[0]["labels"]==1)[0]
    iou=getIouFaster(y[0]["boxes"][doorTruth],door_boxes)

    THRESHOLD_IOU=0.3
    isTp=(iou>=THRESHOLD_IOU)
    isTp=np.sum(isTp,axis=0)
    indexIsTp=np.where(isTp>0)
    indexNotTp=np.where(isTp==0)

    #print(target)
    #print("\n",y)
    #Take out true positive
    boxes=door_boxes[indexIsTp]
    boxes=resizeBoxes(boxes,(1000,1000),(1,1))
    boxes=toList(boxes)
    scores=toList(door_scores[indexIsTp])

    data.append({
        "image_id":y[0]["image_id"],
        "boxes":boxes,
        "labels":1, #1:door  0:not door
        "scores":scores
    })

    #Take out false positive
    boxes=door_boxes[indexNotTp]
    boxes=resizeBoxes(boxes,(1000,1000),(1,1))
    boxes=toList(boxes)
    scores=toList(door_scores[indexNotTp])

    data.append({
        "image_id":y[0]["image_id"],
        "boxes":boxes,
        "labels":0, #1:door  0:not door
        "scores":scores
    })