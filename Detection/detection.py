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


#Set up the model
model=torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model=transfer(model,5)
for param in model.parameters():
        param.requires_grad = True

optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,betas=(0.9, 0.9))
model=train(model,optimizer,labelbox,batch_size=5,epochs=15)

torch.save(model, os.path.join(wandb.run.dir, "model.pt"))

print("finish")
