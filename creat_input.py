import os
import torch
import numpy as np
from tqdm import tqdm

# ============= 配置 =============
input_dir = "/home/mtu/data1/EARI-Polarization-Demosaicking/TokyoTech/Dataset/pth"
save_dir = "/home/mtu/TCPDNet/input_for_TCPDNet"

os.makedirs(save_dir, exist_ok=True)

# ============= 主程序 =============
def process_pth(file_path):
    data = torch.load(file_path)

    # data 形状通常是 [C, H, W]，例如 C=3
    # 如果是 batch 形式（如 [1, 3, H, W]），去掉 batch 维度
    if len(data.shape) == 4:
        data = data[0]

    # 打印形状以确认
    print(f"Loaded {os.path.basename(file_path)} with shape {tuple(data.shape)}")

    # 如果是三通道彩色偏振图（例如 RGB），分别取出
    if data.shape[0] == 3:
        R, G, B = data[0], data[1], data[2]
    else:
        raise ValueError(f"Unexpected channel count: {data.shape[0]}, expected 3 channels (RGB).")

    # ================================
    # 计算 Stokes 参数（S0, u, v）
    # ================================
    # 为演示方便，这里假设输入 data 是一个角度序列的平均值（若你有四角度源文件，可以改为直接取）
    # 我们通过 DoLP (u,v) 的分布合成四个角度下的强度

    # S0 = 总强度
    S0_R = R
    S0_G = G
    S0_B = B

    # 模拟 DoLP (偏振度) 与 AoLP (偏振角)
    # 假设偏振方向为随机均匀分布，这里可以用物理测得的 u,v 替换
    u = torch.zeros_like(R) + 0.5  # DoLP_x 分量
    v = torch.zeros_like(R) + 0.5  # DoLP_y 分量

    # ================================
    # 根据 Stokes 反算四个角度的偏振强度
    # ================================
    def reconstruct_intensity(S0, u, v, theta_deg):
        theta = torch.deg2rad(torch.tensor(theta_deg))
        return 0.5 * (S0 + u * torch.cos(2 * theta) + v * torch.sin(2 * theta))

    R000 = reconstruct_intensity(S0_R, u, v, 0)
    R045 = reconstruct_intensity(S0_R, u, v, 45)
    R090 = reconstruct_intensity(S0_R, u, v, 90)
    R135 = reconstruct_intensity(S0_R, u, v, 135)

    G000 = reconstruct_intensity(S0_G, u, v, 0)
    G045 = reconstruct_intensity(S0_G, u, v, 45)
    G090 = reconstruct_intensity(S0_G, u, v, 90)
    G135 = reconstruct_intensity(S0_G, u, v, 135)

    B000 = reconstruct_intensity(S0_B, u, v, 0)
    B045 = reconstruct_intensity(S0_B, u, v, 45)
    B090 = reconstruct_intensity(S0_B, u, v, 90)
    B135 = reconstruct_intensity(S0_B, u, v, 135)

    # ================================
    # 将RGB三通道转成亮度通道（I = 0.299R + 0.587G + 0.114B）
    # ================================
    I000 = 0.299 * R000 + 0.587 * G000 + 0.114 * B000
    I045 = 0.299 * R045 + 0.587 * G045 + 0.114 * B045
    I090 = 0.299 * R090 + 0.587 * G090 + 0.114 * B090
    I135 = 0.299 * R135 + 0.587 * G135 + 0.114 * B135

    # ================================
    # 拼接为模型输入 [4, H, W]
    # ================================
    CPFA_input = torch.stack([I000, I045, I090, I135], dim=0)

    # 归一化（normalization）
    CPFA_input = CPFA_input / CPFA_input.max()

    # 保存
    filename = os.path.basename(file_path).replace(".pth", "_input4ch.pth")
    save_path = os.path.join(save_dir, filename)
    torch.save(CPFA_input, save_path)

    print(f"✅ Saved {filename} with shape {tuple(CPFA_input.shape)}")


if __name__ == "__main__":
    files = [f for f in os.listdir(input_dir) if f.endswith(".pth")]
    print(f"✅ Found {len(files)} files under {input_dir}\n")

    for f in tqdm(files, desc="Processing"):
        process_pth(os.path.join(input_dir, f))
