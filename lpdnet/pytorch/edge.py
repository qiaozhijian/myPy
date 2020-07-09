import torch
input = torch.range(0,23).reshape(2,3,4)
print(input)
index = torch.tensor([2,1,0,1,0]).reshape(1,1,5).repeat(2,3,1)
print(index.size())
index = index.reshape(2,-1)
output = torch.gather(input,dim=1,index=index)
print(output)