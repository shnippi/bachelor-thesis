import torch

shape = (2,2,2)

tenni = torch.reshape(torch.arange(8), shape)




print(tenni)

# testenni= torch.index_select(tenni,2,torch.tensor([0]))

testenni=tenni.reshape(2,-1).transpose(0,1)

print(testenni)

