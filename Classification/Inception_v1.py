import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T
import torchvision
import numpy as np
from util_torch import*
from torchsummary import summary
NUM_TRAIN = 47000
# The torchvision.transforms package provides tools for preprocessing data
# and for performing data augmentation; here we set up a transform to
# preprocess the data by subtracting the mean RGB value and dividing by the
# standard deviation of each RGB value; we've hardcoded the mean and std.
transform = T.Compose([
                T.Resize((151,151)),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

# We set up a Dataset object for each split (train / val / test); Datasets load
# training examples one at a time, so we wrap each Dataset in a DataLoader which
# iterates through the Dataset and forms minibatches. We divide the CIFAR-10
# training set into train and val sets by passing a Sampler object to the
# DataLoader telling how it should sample from the underlying Dataset.
cifar10_train = dset.CIFAR10('./Datasets', train=True, download=True,
                             transform=transform)
loader_train = DataLoader(cifar10_train, batch_size=128, 
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

cifar10_val = dset.CIFAR10('./Datasets', train=True, download=True,
                           transform=transform)
loader_val = DataLoader(cifar10_val, batch_size=128, 
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

cifar10_test = dset.CIFAR10('./Dtasets', train=False, download=True, 
                            transform=transform)
loader_test = DataLoader(cifar10_test, batch_size=128)

#model = torch.hub.load('pytorch/vision:v0.5.0', 'inception_v3', pretrained=True,progress=True)
models={
        #"InceptionV1":torchvision.models.googlenet(pretrained=True),
        #"Resnet34":torchvision.models.resnet34(pretrained=True),
        "Resnet50":torchvision.models.resnet50(pretrained=False),
        #"Resnet101":torchvision.models.resnet101(pretrained=True),
        #"InceptionV3":torchvision.models.inception_v3(pretrained=True)
        }   
for k,model in models.items():
    with open("Inception.log",'a') as file:
        file.write("\n**********"+k+"***********\n")
        file.close()
    print("********** ",k,"*********")
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 10)

    for param in model.parameters():
        param.requires_grad = True
    print("Model Ready")

    optimizer = optim.SGD(model.parameters(), lr=1e-2,weight_decay=1e-6,momentum=0.9)
    #optimizer=optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999),weight_decay=1e-4)
    train(k,model,optimizer,loader_train,loader_val,20)

    test_acc=check_accuracy(loader_test,model)
    print(k," Test_acc:%.3f"%(test_acc*100),"%")
    with open("Inception.log",'a') as file:
        file.write(k+"Test_acc: "+str(test_acc*100)+"%\n\n")
        file.close()
    