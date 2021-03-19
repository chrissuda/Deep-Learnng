import torch
import torch.nn as nn
import torch.nn.functional as F

class Position(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2,num_classes):
        super().__init__()
        # assign layer objects to class attributes
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        nn.init.kaiming_normal_(self.fc1.weight)
        #self.bn1=torch.nn.BatchNorm1d(hidden_size_1)

        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        nn.init.kaiming_normal_(self.fc2.weight)
        #self.bn2=torch.nn.BatchNorm1d(hidden_size_2)

        self.fc3 = nn.Linear(hidden_size_2,num_classes)
        nn.init.kaiming_normal_(self.fc3.weight)
        # self.bn3=torch.nn.BatchNorm1d(hidden_size_3)


    def forward(self, x):
        # forward always defines connectivity
        x = F.relu(self.fc1(x))
        #x=self.bn1(x)

        x = F.relu(self.fc2(x))
        #x=self.bn2(x)

        x = self.fc3(x)
        
        return x