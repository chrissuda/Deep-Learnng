import torch
import torchvision
from torchvision import transforms
from Loader import*
from CocoDataset import labelboxCoco
from util_labelbox import count
from util_detection import*
from util_train import*

NUM_VAL=2
model_path="../model.pt"
batch_size=1
annFile="NYCCoco.json"
img_root="/home/students/cnn/NYC_PANO"
newSize=(1500,1500)
epochs=3

# Using GPU or CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cuda.cufft_plan_cache.clear()
else:
    device = torch.device('cpu')

transform = transforms.Compose([
                transforms.Resize(newSize),
                transforms.ToTensor()])

labelbox=labelboxCoco(img_root,annFile,newSize,transform=transform)
print("Dataset size:",len(labelbox))
loader_train=Loader(labelbox,end=len(labelbox)-NUM_VAL,batch_size=batch_size,shuffle=True)
loader_val=Loader(labelbox,start=len(labelbox)-NUM_VAL,batch_size=batch_size,shuffle=False)

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
# checkAp(model,loader_val,device,NMS=True)


for i,(x, y) in enumerate(loader_val):
    # move to device, e.g. GPU
    x=x.to(device=device, dtype=torch.float32)  
    target = model(x)
    try:    
        target[0]["scores"],target[0]["labels"],target[0]["boxes"]=nms(target[0]["boxes"],target[0]["labels"],target[0]["scores"],0.4)

        predict_file="../experiment_nyc/"+y[0]["image_id"][:-4]+"_predict.jpg"
        truth_file="../experiment_nyc/"+y[0]["image_id"][:-4]+"_truth.jpg"
        draw(x[0],target[0],file=predict_file)
        draw(x[0],y[0],file=truth_file)
        
    except:
        print("\n\ni:",i," image_id:",y[0]["image_id"],"\n\n")

