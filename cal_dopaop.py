import os
import numpy as np

# ===============================
# 输入路径
# ===============================
input_dir = "/home/mtu/TCPDNet/output_TCPDNet/apple_input4ch"

# 偏振角文件
angles = ["P000", "P045", "P090", "P135"]
files = [os.path.join(input_dir, f"{angle}_Intensity.npy") for angle in angles]

# ===============================
# 读取Intensity数据并归一化到0~255
# ===============================
I_list = []
for f in files:
    I = np.load(f).astype(np.float32)
    # 如果原始数据范围是0~1，则放大到0~255
    if I.max() <= 1.0:
        I = I * 255.0
    I_list.append(I)

I0, I45, I90, I135 = I_list

# ===============================
# 计算Stokes参数
# ===============================
# S0 = I0 + I90
S0 = I0 + I90
# S1 = I0 - I90
S1 = I0 - I90
# S2 = I45 - I135
S2 = I45 - I135

# ===============================
# 计算DoLP与AoLP
# ===============================
DoLP = np.sqrt(S1**2 + S2**2) / (S0 + 1e-8)  # 避免除零
AoLP = 0.5 * np.arctan2(S2, S1) * (180 / np.pi)  # 转换为角度

# 将AoLP范围映射到0~180°
AoLP = np.mod(AoLP, 180.0)

# ===============================
# 输出统计信息
# ===============================
print("DoLP statistics:")
print(f" min: {DoLP.min():.4f}, max: {DoLP.max():.4f}, mean: {DoLP.mean():.4f}")

print("\nAoLP statistics (deg):")
print(f" min: {AoLP.min():.2f}, max: {AoLP.max():.2f}, mean: {AoLP.mean():.2f}")


