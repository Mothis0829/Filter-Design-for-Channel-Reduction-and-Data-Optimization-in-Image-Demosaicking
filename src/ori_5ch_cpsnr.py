import os
import torch
import math
import h5py

# ================== 路径设置 ==================
GT_DIR   = r"C:\mtu\mtu\TCPDNet\dataset\TokyoTech\hdf5"
PRED_DIR = r"C:\mtu\mtu\TCPDNet\dataset\TokyoTech\hdf5(mo)"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-12


# ================== CPSNR ==================
def CPSNR(x, y, peak=1.0, b=15):
    """
    x, y: [1, H, W]
    """
    x = x[..., b:-b, b:-b]
    y = y[..., b:-b, b:-b]
    mse = torch.mean((x - y) ** 2)
    if mse < EPS:
        return torch.tensor(100.0, device=x.device)
    return 10 * torch.log10(peak ** 2 / mse)


# ================== 12ch → SR0,SG0,SB0,u,v ==================
def calc_rgb0_uv(R0, R45, R90, R135,
                 G0, G45, G90, G135,
                 B0, B45, B90, B135):

    SR0 = (R0 + R45 + R90 + R135) / 2
    SG0 = (G0 + G45 + G90 + G135) / 2
    SB0 = (B0 + B45 + B90 + B135) / 2

    denom = SR0 + SG0 + SB0 + 1e-8
    u = ((R0 + G0 + B0) - (R90 + G90 + B90)) / denom
    v = ((R45 + G45 + B45) - (R135 + G135 + B135)) / denom

    return SR0, SG0, SB0, u, v


# ================== u,v → DoP / AoP ==================
def calc_dop_aop(u, v):
    dop = torch.sqrt(u ** 2 + v ** 2)
    aop = 0.5 * torch.atan2(v, u)  # rad
    return dop, aop


# ================== 加载 hdf5 并拆 12 通道 ==================
def load_12ch_hdf5(hdf5_path, key=None):
    """
    支持的数据格式：
    - [12, H, W]
    - [4, 3, H, W]
    """

    with h5py.File(hdf5_path, "r") as f:
        if key is None:
            key = list(f.keys())[0]   # 默认取第一个 dataset
        data = f[key][:]

    data = torch.from_numpy(data).float().to(device)

    # [12,H,W]
    if data.dim() == 3 and data.shape[0] == 12:
        R0, R45, R90, R135 = data[0:4]
        G0, G45, G90, G135 = data[4:8]
        B0, B45, B90, B135 = data[8:12]

    # [4,3,H,W]
    elif data.dim() == 4 and data.shape[0] == 4:
        R0, R45, R90, R135 = data[:, 0]
        G0, G45, G90, G135 = data[:, 1]
        B0, B45, B90, B135 = data[:, 2]

    else:
        raise ValueError(f"Unsupported HDF5 shape: {data.shape}")

    return R0, R45, R90, R135, G0, G45, G90, G135, B0, B45, B90, B135


# ================== 主评估流程 ==================
def evaluate():
    files = sorted(f for f in os.listdir(GT_DIR) if f.endswith(".hdf5"))
    assert files, "GT directory is empty"

    cpsnr_s0_sum = torch.zeros(3, device=device)  # SR0 SG0 SB0
    cpsnr_dop_sum = 0.0
    cpsnr_aop_sum = 0.0
    count = 0

    for name in files:
        gt_path   = os.path.join(GT_DIR, name)
        pred_path = os.path.join(PRED_DIR, name)

        if not os.path.exists(pred_path):
            print(f"[Skip] Missing pred: {name}")
            continue

        gt   = load_12ch_hdf5(gt_path)
        pred = load_12ch_hdf5(pred_path)

        SR0_g, SG0_g, SB0_g, u_g, v_g = calc_rgb0_uv(*gt)
        SR0_p, SG0_p, SB0_p, u_p, v_p = calc_rgb0_uv(*pred)

        dop_g, aop_g = calc_dop_aop(u_g, v_g)
        dop_p, aop_p = calc_dop_aop(u_p, v_p)

        # ===== S0 RGB CPSNR =====
        cpsnr_s0_sum[0] += CPSNR(SR0_p.unsqueeze(0), SR0_g.unsqueeze(0))
        cpsnr_s0_sum[1] += CPSNR(SG0_p.unsqueeze(0), SG0_g.unsqueeze(0))
        cpsnr_s0_sum[2] += CPSNR(SB0_p.unsqueeze(0), SB0_g.unsqueeze(0))

        # ===== DoP / AoP CPSNR =====
        cpsnr_dop_sum += CPSNR(dop_p.unsqueeze(0), dop_g.unsqueeze(0))
        cpsnr_aop_sum += CPSNR(aop_p.unsqueeze(0), aop_g.unsqueeze(0))

        count += 1

    cpsnr_s0 = cpsnr_s0_sum / count
    cpsnr_dop = cpsnr_dop_sum / count
    cpsnr_aop = cpsnr_aop_sum / count

    print("\n===== CPSNR Results =====")
    print(f"CPSNR/SR0 : {cpsnr_s0[0]:.2f} dB")
    print(f"CPSNR/SG0 : {cpsnr_s0[1]:.2f} dB")
    print(f"CPSNR/SB0 : {cpsnr_s0[2]:.2f} dB")
    print(f"CPSNR/S0 Avg : {torch.mean(cpsnr_s0):.2f} dB")
    print(f"CPSNR/DoP : {cpsnr_dop:.2f} dB")
    print(f"CPSNR/AoP : {cpsnr_aop:.2f} dB")


# ================== Run ==================
if __name__ == "__main__":
    evaluate()


