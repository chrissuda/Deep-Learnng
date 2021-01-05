import torch

a=torch.tensor([[1,2,3,4]])
a[:,(1,3)]*=2
print(a)