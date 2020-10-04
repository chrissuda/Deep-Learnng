import torch
import torchvision
from torchvision import transforms
from Loader import*
from CocoDataset import labelboxCoco
from util_labelbox import count
from util_detection import*
from util_train import*


NUM_VAL=7
model_path="../model.pt"
batch_size=3
annFile="labelboxCoco.json"
root="../images"
newSize=(800,800)
device = torch.device('cuda')
epochs=5

transform = transforms.Compose([
                transforms.Resize(newSize),
                transforms.ToTensor()])

labelbox=labelboxCoco(root,annFile,newSize,transform=transform)
loader_train=Loader(labelbox,end=len(labelbox)-NUM_VAL,batch_size=batch_size,shuffle=True)
loader_val=Loader(labelbox,start=200,end=200+NUM_VAL,batch_size=batch_size,shuffle=False)

# count(loader_val)
# test(loader_val)

#Set up the model
# model=torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# #4 categories + 1 background
# model=transfer(model,5)

# for param in model.parameters():
#         param.requires_grad = True



# optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,betas=(0.9, 0.95))
# model=train(model,optimizer,epochs,
#                 loader_train=loader_train,loader_val=loader_val,wb=False)


# torch.save(model,model_path)

model=torch.load(model_path)
folder="../NYC"
predictOnImageFolder(folder,model_path,0.3)
predictOnImageFolder(folder,model_path,0.3,NMS=True)
#checkAp(model,loader_val)
print("finish")
