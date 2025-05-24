import torch, platform
print(torch.backends.mps.is_available(), platform.machine())