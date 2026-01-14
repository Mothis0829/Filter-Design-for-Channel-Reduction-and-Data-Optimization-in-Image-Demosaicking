import torch
import os

pth_path = "/home/mtu/TCPDNet/input_for_TCPDNet/yarn_input4ch.pth"

# 加载
data = torch.load(pth_path, map_location='cpu')

# 如果是字典
if isinstance(data, dict):
    print("Keys in dict:", data.keys())
    for k, v in data.items():
        print(f"\nKey: {k}")
        if isinstance(v, torch.Tensor):
            print("Tensor shape:", v.shape)
            print("Sample values (top-left 5x5):")
            print(v.flatten()[:25].reshape(5,5))
        else:
            print(v)
# 如果直接是 Tensor
elif isinstance(data, torch.Tensor):
    print("Tensor shape:", data.shape)
    print("Sample values (top-left 5x5):")
    print(data.flatten()[:25].reshape(5,5))

