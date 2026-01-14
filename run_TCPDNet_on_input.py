import os
import torch
import sys
import numpy as np

# 添加 src 路径
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.CPFA import CPFA
from src.CPFANetBase import CPFANetBase

# 路径配置
input_dir = "/home/mtu/TCPDNet/input_for_TCPDNet"
output_dir = "/home/mtu/TCPDNet/output_TCPDNet"
os.makedirs(output_dir, exist_ok=True)

pretrained_path = "/home/mtu/TCPDNet/checkpoints/net_00000200.pt"

# 配置（根据你的训练 config.yaml 设置）

config = {
    "net_approach": "TwoStepColorPolarNet",
    "color_net": {
        "method": "ColorBilinearRefine",
        "net_base": "/home/mtu/TCPDNet/src/configs/base_net/unet_default.yaml",
        "skip_connection_flag": True
    },
    "polar_net": {
        "method": "PolarBilinearRefine",
        "net_base": "/home/mtu/TCPDNet/src/configs/base_net/unet_default.yaml",
        "skip_connection_flag": True
    }
}

# 设备配置

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 构建 CPFA

cpfa = CPFA(color_order="gbrg")
cpfa.to(device)

# 构建网络

model = CPFANetBase.build_net(config, cpfa)
model.to(device)
model.eval()

# 加载预训练权重
checkpoint = torch.load(pretrained_path, map_location=device)
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
else:
    model.load_state_dict(checkpoint, strict=False)

print("✅ Loaded pretrained model (strict=False).")

# 遍历输入文件进行推理

input_files = [f for f in os.listdir(input_dir) if f.endswith(".pth")]
if not input_files:
    raise ValueError(f"No .pth files found in {input_dir}")

print(f"✅ Found {len(input_files)} files under {input_dir}")

def process_pth(input_path):
    # 加载输入 (4, H, W)
    data = torch.load(input_path, map_location=device)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]

    data = data.to(dtype=torch.float32, device=device)

    if data.ndim == 3:
        data = data.unsqueeze(0)  # batch 维

    if data.shape[1] != 4:
        raise ValueError(f"Input shape mismatch: expected 4 channels, got {data.shape}")

    # 生成子阵列
    sub_bayers = cpfa.sparse_sub_bayer(data)

    # 模型推理

    with torch.no_grad():
        sub_colors, full_color_polar = model(sub_bayers)

    # 输出 Intensity 矩阵 & 保存

    intensity_dir = os.path.join(output_dir, os.path.basename(input_path)[:-4])
    os.makedirs(intensity_dir, exist_ok=True)

    angles = ["P000", "P045", "P090", "P135"]
    intensity_summary = {}

    for i, angle in enumerate(angles):
        I_matrix = full_color_polar[0, i, :, :]  # (H, W)
        I_matrix_255 = (I_matrix * 255).cpu().numpy()  # 转到 [0,255] 并转换为 numpy

        # 保存矩阵为 .npy
        np.save(os.path.join(intensity_dir, f"{angle}_Intensity.npy"), I_matrix_255)

        # 计算总强度和平均强度
        I_total = I_matrix_255.sum()
        I_mean  = I_matrix_255.mean()
        intensity_summary[angle] = {"total": I_total, "mean": I_mean}

        print(f"✅ Saved Intensity for {angle} (shape: {I_matrix_255.shape})")
        print(f"   Total: {I_total:.2f}, Mean: {I_mean:.2f}")

    # 保存全部结果为 PTH
    output_pth_path = os.path.join(output_dir, os.path.basename(input_path))
    torch.save({
        "sub_colors": sub_colors.cpu(),
        "full_color_polar": full_color_polar.cpu(),
        "intensity_summary": intensity_summary
    }, output_pth_path)

    print(f"✅ Saved PTH and Intensity data for {os.path.basename(input_path)}\n")

# 循环处理所有输入文件

for file in sorted(input_files):
    process_pth(os.path.join(input_dir, file))

print("🎉 All files processed.")


