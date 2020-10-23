import torch
import torchvision
import json
from torchvision import transforms
from Loader import*
from CocoDataset import labelboxCoco
from util_labelbox import count
from util_detection import*
from util_train import*

base_model_path="../base_model.pt"
model_path="../model.pt"
batch_size=8
train_annFile="Train.json"
val_annFile="Val.json"

img_root="/home/students/cnn/NYC_PANO"
newSize=(1000,1000)
epochs=1

# Using GPU or CPU
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
print(len(loader_train))
val_labelbox=labelboxCoco(img_root,val_annFile,newSize,transform=transform)
loader_val=Loader(val_labelbox,batch_size=batch_size)
print(len(loader_val))

# count(loader_val)
# test(loader_val)

#Set up the model
# model=torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# #4 categories + 1 background
# model=transfer(model,5)



model=torch.load(model_path,map_location=device)
# for param in model.parameters():
#         param.requires_grad = True

# optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,betas=(0.9, 0.95))
# model=train(model,optimizer,epochs,
#                 loader_train=loader_train,loader_val=loader_val,device=device,wb=False)

# torch.save(model,model_path)


# folder="../NYC"
# predictOnImageFolder(folder,model_path,0.3)
# predictOnImageFolder(folder,model_path,0.3,NMS=True)
model.eval()
loader_val=Loader(val_labelbox,batch_size=batch_size)
checkAp(model,loader_val,device,NMS=True,Filter=True)

# loader_val=Loader(val_labelbox,batch_size=1)
# for i,(x, y) in enumerate(loader_val):
#     try:
#         # move to device, e.g. GPU
#         x=x.to(device=device, dtype=torch.float32)
#         target = model(x)
       
#         target[0]["scores"],target[0]["labels"],target[0]["boxes"]=nms(target[0]["boxes"],target[0]["labels"],target[0]["scores"],0.4)
#         target[0]["scores"],target[0]["labels"],target[0]["boxes"]=snms(target[0]["boxes"],target[0]["labels"],target[0]["scores"],0.1)
#         predict_file="../experiment_nyc/"+y[0]["image_id"][:-4]+"_predict_1.jpg"
#         truth_file="../experiment_nyc/"+y[0]["image_id"][:-4]+"_truth.jpg"
#         draw(x[0],target[0],file=predict_file)
#         draw(x[0],y[0],file=truth_file)
        
#     except Exception as e:
#         print("\ni:",i," image_id:",y[0]["image_id"],e,"\n")

