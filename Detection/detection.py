import torch
import torchvision
from torchvision import transforms
from Loader import*
from CocoFormat import labelboxCoco
from util_labelbox import count
from util_detection import*
from util_train import*


NUM_VAL=100
model_path="../model.pt"
batch_size=8
annFile="labelboxCoco.json"
root="../images"
newSize=(800,800)

transform = transforms.Compose([
                transforms.Resize(newSize),
                transforms.ToTensor()])

labelbox=labelboxCoco(root,annFile,newSize,transform=transform)
loader_train=Loader(labelbox,end=len(labelbox)-NUM_VAL,batch_size=batch_size,shuffle=True)
loader_val=Loader(labelbox,start=len(labelbox)-NUM_VAL,batch_size=batch_size,shuffle=True)

# count(loader_val)
# test(loader_val)

#Set up the model
model=torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

model=transfer(model,5)
#model.roi_heads.box_head = torch.nn.Sequential(torch.nn.Flatten(),
#                                                model.roi_heads.box_head.fc6,
#                                                torch.nn.ReLU(),torch.nn.Dropout())

print(model)
for param in model.parameters():
        param.requires_grad = True

for param in model.backbone.parameters():
        param.requires_grad=False

# for param in model.backbone[:-2].parameters():
#     param.requires_grad = False

# for name,param in model.named_parameters():
#     if param.requires_grad==False:
#        print(name)

optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,betas=(0.9, 0.95))
model=train(model,optimizer,10,
                loader_train=loader_train,loader_val=loader_val,wb=True)

torch.save(model,"../original.pt")


print("finish")
