import torch
import torchvision
from torchvision import transforms
from util_detection import*
from CocoFormat import labelboxCoco

annFile="labelboxCoco.json"
root="../images"
newSize=(800,800)


transform = transforms.Compose([
                transforms.Resize(newSize),
                transforms.ToTensor()])

labelbox=labelboxCoco(root,annFile,newSize,transform=transform)
train_loader=loader(labelbox,5)

#Set up the model
model=torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model=transfer(model,5)
for param in model.parameters():
        param.requires_grad = True


optimizer=torch.optim.Adam(model.parameters(), lr=0.001)
model=train(model,optimizer,train_loader,epochs=2)
print("finish")
