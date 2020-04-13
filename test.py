from itertools import chain
import torch
import torchvision
import torchvision.transforms as T
from Coco import *
import json
from tqdm import tqdm

def test():
    root="./labelbox_img"
    annFile="labelboxCoco.json"

    transform = T.Compose([
                    T.Resize((800,800)),
                    T.ToTensor()])

    labelbox=labelboxCoco(root,annFile,transform=transform)

    model=torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    iou,confidence=[],[]
    doorNum=0;
    total=0
    labelbox_loader=loader(labelbox,3)
    model.eval()
    bar=tqdm(total=489)
    for x,y in labelbox_loader:
        predict=model(x)
        
        for j in range(len(predict)): #Coresponding to an image
            #Turn torch tensor into list
            y[j]["boxes"]=y[j]["boxes"].tolist()
            y[j]["labels"]=y[j]["labels"].tolist()
            predict[j]["boxes"]=predict[j]["boxes"].tolist()
            predict[j]["labels"]=predict[j]["labels"].tolist()
            predict[j]["scores"]=predict[j]["scores"].tolist()

            label=predict[j]["labels"]
            index_predict=[i for i in range(len(label)) if label[i] == 71]

            index_truth=[i for i in range(len(y[j]["labels"])) if y[j]["labels"][i] == 71]
            doorNum+=len(index_predict)
            print(index_truth)
            if(len(index_predict)!=0 and len(index_truth!=0)):
                box=predict[j]["boxes"]
                box_predict=[box[i] for i in index_predict]
                box_truth=[y[j]["boxes"][i] for i in index_truth]
                box_index=set() #get the index where there is Door based on truth and prediction
                for b in range(len(box_truth)):
                    box_result=[]
                    for bb in range(len(box_predict)):
                        box_result.append(IoU(box_truth[b],box_predict[bb]))    
                    box_max=max(box_result)
                    box_index.add(box_result.index(box_max))
                    iou.append(box_max)

                score=predict[j]["scores"]
                score=[score[i] for i in box_index]
                confidence.append(score)

        total+=1;
        bar.update(3)

    iouList=list(chain(*iou))
    print("average_iou:",sum(iouList)/len(iouList))
    confidenceList=list(chain(*iou))
    print("average_confidence:",sum(confidenceList)/len(confidenceList))
    print("Total Door:",489," Predction:",doorNum)
    print("Prediction_accuracy:%.2f%",float(doorNum)*100.0/489)
    print("Total iteration:",total)
    dict={"iouList":iouList,"confidenceList":confidenceList}
    with open("experiment.json","w") as f:
        json.dump(dict,f)

test()
