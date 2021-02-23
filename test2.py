import torch

print(torch.randn(5).cuda())

print(torch.cuda.get_device_name())

print(torch.cuda.memory_allocated())