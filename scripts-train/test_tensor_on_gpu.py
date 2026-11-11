import torch

x = torch.randn(1000, 1000, device="cuda")
y = x @ x

print(y.device)
print("Allocated MB:", torch.cuda.memory_allocated() / 1024**2)