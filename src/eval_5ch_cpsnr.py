import torch
import torch.nn as nn
import numpy as np

from utils import (
    get_config,
    get_all_data_loaders,
    get_CPFA,
    get_model_name_from_config_path,
    get_model_list
)
from CPFANetBase import CPFANetBase
from evals import filter_5channels
from trainer import *
import argparse
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ------------------- CPSNR 定义 -------------------
def CPSNR(x, y, peak=1, b=15):
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if y.dim() == 3:
        y = y.unsqueeze(0)

    x = x[:, :, b-1:x.shape[2]-b, b-1:x.shape[3]-b]
    y = y[:, :, b-1:y.shape[2]-b, b-1:y.shape[3]-b]

    mse = nn.MSELoss(reduction="none")
    batch_psnr = mse(x, y).detach()
    cpsnr = torch.mean(
        10 * torch.log10(peak**2 / (torch.mean(batch_psnr, (1,2,3)) + 1e-32))
    )
    return cpsnr


# ------------------- DoP / AoP 计算 -------------------
def compute_dop_aop(y_5):
    """
    y_5: [B, 5, H, W] = [SR0, SG0, SB0, u, v]
    """
    u = y_5[:, 3:4, :, :]
    v = y_5[:, 4:5, :, :]

    dop = torch.sqrt(u ** 2 + v ** 2 + 1e-8)
    aop = 0.5 * torch.atan2(v, u)  # radian

    return dop, aop


# ------------------- 5ch + DoP + AoP CPSNR -------------------
def eval_net_5ch(net, loader):
    net.eval()

    # SR0, SG0, SB0, u, v, DoP, AoP
    cpsnr_sum = torch.zeros(7, device=device)
    count = 0

    with torch.no_grad():
        for idx, raw_CPFA, gt, _ in loader:
            raw_CPFA = raw_CPFA.to(device)
            gt = gt.to(device)

            result = net(raw_CPFA)
            if isinstance(result, tuple):
                _, y = result
            else:
                y = result

            y_5 = filter_5channels(y)
            gt_5 = filter_5channels(gt)

            B = y_5.shape[0]
            count += B

            # ---- 5 channels ----
            for c in range(5):
                cpsnr_c = CPSNR(
                    y_5[:, c:c+1, :, :],
                    gt_5[:, c:c+1, :, :]
                )
                cpsnr_sum[c] += cpsnr_c * B

            # ---- DoP / AoP ----
            pred_dop, pred_aop = compute_dop_aop(y_5)
            gt_dop, gt_aop = compute_dop_aop(gt_5)

            cpsnr_sum[5] += CPSNR(pred_dop, gt_dop) * B
            cpsnr_sum[6] += CPSNR(pred_aop, gt_aop) * B

    cpsnr_avg = cpsnr_sum / count

    names = ["SR0", "SG0", "SB0", "u", "v", "DoP", "AoP"]
    print("====== CPSNR (5ch + DoP + AoP) ======")
    for i, name in enumerate(names):
        print(f"CPSNR/{name}: {cpsnr_avg[i].item():.2f} dB")

    print(f"CPSNR/Average (5ch): {torch.mean(cpsnr_avg[:5]).item():.2f} dB")
    print(f"CPSNR/Average (Polar): {torch.mean(cpsnr_avg[3:7]).item():.2f} dB")

    return cpsnr_avg


# ------------------- model loading -------------------
def load_model(cfg_path):
    config = get_config(cfg_path)
    train_loader, trainv_loader, val_loader, test_loader = get_all_data_loaders(config)
    cpfa = get_CPFA(config)
    cpfa.to(device)
    net = CPFANetBase.build_net(config, cpfa).to(device)
    return net, test_loader, config


def best_eval_5ch(cfg_path):
    net, test_loader, config = load_model(cfg_path)
    model_name = get_model_name_from_config_path(cfg_path)

    if 'opt_approach' in config:
        output_directory = os.path.join("outputs", model_name)
        ckp_dir = os.path.join(output_directory, "checkpoints")
        trainer = globals()[config['opt_approach']](net, ckp_dir, config)
        trainer.load_best(config, True)
        trainer.to(device)
        trainer.eval()
    else:
        net.eval()

    eval_net_5ch(net, test_loader)


def avg_5_models_5ch(cfg_path):
    base_ckp_dir = r"C:\mtu\mtu\TCPDNet\checkpoints"
    net, test_loader, config = load_model(cfg_path)
    model_name = get_model_name_from_config_path(cfg_path)

    if 'opt_approach' not in config:
        raise ValueError('avg_5_models_5ch is not applicable.')

    keys = ["SR0", "SG0", "SB0", "u", "v", "DoP", "AoP", "Average_5ch", "Average_Polar"]
    cpsnr_list = {k: [] for k in keys}

    for i in range(1, 6):
        tail = "_T_" + str(i)
        ckp_dir = os.path.join(base_ckp_dir, model_name + tail, "checkpoints")

        trainer = globals()[config['opt_approach']](net, ckp_dir, config)
        checkpoint = get_model_list(ckp_dir, "net")[-1]
        trainer.resume(config, checkpoint, not torch.cuda.is_available())
        trainer.to(device)
        trainer.eval()

        cpsnr = eval_net_5ch(net, test_loader).cpu().numpy()

        for j, k in enumerate(["SR0", "SG0", "SB0", "u", "v", "DoP", "AoP"]):
            cpsnr_list[k].append(cpsnr[j])

        cpsnr_list["Average_5ch"].append(np.mean(cpsnr[:5]))
        cpsnr_list["Average_Polar"].append(np.mean(cpsnr[3:7]))

    print("\n====== 5-model Average CPSNR (5ch + DoP + AoP) ======")
    for k, v in cpsnr_list.items():
        print(f"{k}: {np.mean(v):.2f} ± {np.std(v):.2f}")


# ------------------- main -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument(
        '--mode',
        type=str,
        default='best',
        choices=['best', 'avg_5_models']
    )
    args = parser.parse_args()

    if args.mode == 'best':
        best_eval_5ch(args.config)
    else:
        avg_5_models_5ch(args.config)


