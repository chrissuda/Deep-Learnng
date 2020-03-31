
import torch 

d=torch.tensor([[1,2,3],[4,5,6]])
e=torch.tensor([[7,8,9],[10,11,12]])
a=torch.tensor([1,0,4,4])
b=torch.tensor([0,0,3,4])
print(torch.cat((d,e)))