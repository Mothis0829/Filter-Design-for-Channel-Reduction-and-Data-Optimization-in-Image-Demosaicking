# =========================
# evals.py (FULL VERSION)
# =========================

from dataset import BaseTestingDataset, PatchDataset
import torch
from metrics import eval_all
import os
import numpy as np
from utils import (
    get_config, get_all_data_loaders, get_CPFA,
    get_model_list, get_model_name_from_config_path
)
from CPFANetBase import CPFANetBase
from loss import LossL1

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# 5-channel filter & reconstruction
# -------------------------------------------------

def filter_5channels(y):
    """
    y: [B,4,3,H,W]
    return: [B,5,H,W] -> SR0,SG0,SB0,u,v
    """
    B, _, _, H, W = y.shape
    y5 = torch.zeros(B, 5, H, W, device=y.device)

    SR0 = (y[:,0,0] + y[:,1,0] + y[:,2,0] + y[:,3,0]) / 2
    SG0 = (y[:,0,1] + y[:,1,1] + y[:,2,1] + y[:,3,1]) / 2
    SB0 = (y[:,0,2] + y[:,1,2] + y[:,2,2] + y[:,3,2]) / 2

    denom = SR0 + SG0 + SB0 + 1e-8

    u = ((y[:,0].sum(1)) - (y[:,2].sum(1))) / denom
    v = ((y[:,1].sum(1)) - (y[:,3].sum(1))) / denom

    y5[:,0] = SR0
    y5[:,1] = SG0
    y5[:,2] = SB0
    y5[:,3] = u
    y5[:,4] = v

    return y5


def reconstruct_12channels_from_5(y5):
    """
    y5: [B,5,H,W]
    return: [B,4,3,H,W]
    """
    B, _, H, W = y5.shape
    y12 = torch.zeros(B, 4, 3, H, W, device=y5.device)

    SR0, SG0, SB0, u, v = y5[:,0], y5[:,1], y5[:,2], y5[:,3], y5[:,4]

    for c, S in enumerate([SR0, SG0, SB0]):
        y12[:,0,c] = S/2 * (1 + u)
        y12[:,2,c] = S/2 * (1 - u)
        y12[:,1,c] = S/2 * (1 + v)
        y12[:,3,c] = S/2 * (1 - v)

    return y12

# -------------------------------------------------
# DOP / AOP computation
# -------------------------------------------------

def compute_dop_aop_from_12ch(y):
    """
    y: [B,4,3,H,W]
    return:
        dop: [B,H,W]
        aop: [B,H,W] (rad)
    """
    I0   = y[:,0].sum(1)
    I45  = y[:,1].sum(1)
    I90  = y[:,2].sum(1)
    I135 = y[:,3].sum(1)

    S0 = I0 + I90
    S1 = I0 - I90
    S2 = I45 - I135

    dop = torch.sqrt(S1**2 + S2**2) / (S0 + 1e-8)
    aop = 0.5 * torch.atan2(S2, S1)

    return dop, aop

# -------------------------------------------------
# Error map visualization
# -------------------------------------------------

def save_error_map(err, path, title):
    plt.figure(figsize=(6,5))
    im = plt.imshow(err, cmap="jet")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

# -------------------------------------------------
# Evaluation
# -------------------------------------------------

def eval_net(
    net,
    loader,
    writer=None,
    tag=None,
    epoch=None,
    pattern_sp=None,
    save_errormap=False,
    errormap_dir="errormaps"
):
    print("eval_net called, save_errormap =", save_errormap)

    if save_errormap:
        os.makedirs(errormap_dir, exist_ok=True)

    net.eval()
    avg_metrics = {}
    l1_loss = 0
    l1_loss_est = LossL1(torch.mean, axis=1)

    if save_errormap:
        os.makedirs(errormap_dir, exist_ok=True)

    for idx, raw_CPFA, gt, _ in loader:
        raw_CPFA = raw_CPFA.to(device)
        gt = gt.to(device)

        with torch.no_grad():
            result = net(raw_CPFA)
            y = result[1] if isinstance(result, tuple) else result

            # -------- filter & reconstruct --------
            y5 = filter_5channels(y)
            y = reconstruct_12channels_from_5(y5)

            # -------- metrics --------
            metrics = eval_all(y, gt)
            for k, v in metrics.items():
                avg_metrics[k] = avg_metrics.get(k, 0) + v.item() * len(raw_CPFA)

            l1_loss += l1_loss_est(y, gt).mean() * len(raw_CPFA)

            # -------- errormap --------
            if save_errormap:
                dop_p, aop_p = compute_dop_aop_from_12ch(y)
                dop_g, aop_g = compute_dop_aop_from_12ch(gt)

                dop_err = torch.abs(dop_p - dop_g)

                aop_diff = torch.abs(aop_p - aop_g)
                aop_err = torch.minimum(aop_diff, np.pi - aop_diff)

                save_error_map(
                    dop_err[0].cpu().numpy(),
                    os.path.join(errormap_dir, f"batch_{idx}_dop_error.jpg"),
                    "DOP Error Map"
                )

                save_error_map(
                    aop_err[0].cpu().numpy(),
                    os.path.join(errormap_dir, f"batch_{idx}_aop_error.jpg"),
                    "AOP Error Map (rad)"
                )

    N = len(loader.dataset)
    for k in avg_metrics:
        avg_metrics[k] /= N
        print(f"{k}: {avg_metrics[k]}")

    print(f"L1 loss: {l1_loss / N}")

    return avg_metrics


# -------------------------------------------------
# Model loading & running
# -------------------------------------------------

import argparse
from trainer import *

def load_model(cfg):
    config = get_config(cfg)
    _, _, _, test_loader = get_all_data_loaders(config)
    cpfa = get_CPFA(config).to(device)
    net = CPFANetBase.build_net(config, cpfa).to(device)
    return net, test_loader, config

def avg_5_models(cfg):
    base = r"C:\mtu\mtu\TCPDNet\checkpoints"

    config = get_config(cfg)
    name = get_model_name_from_config_path(cfg)

    # 只取 test_loader（和训练保持一致）
    _, _, _, test_loader = get_all_data_loaders(config)

    for i in range(1, 6):
        print(f"\n===== Evaluating model T_{i} =====")

        # ---------- 1. 每个模型都重新构建 CPFA + Net ----------
        cpfa = get_CPFA(config).to(device)
        net = CPFANetBase.build_net(config, cpfa).to(device)

        # ---------- 2. checkpoint 路径（⭐ 你刚才这里丢了） ----------
        ckpt_dir = os.path.join(
            base,
            f"{name}_T_{i}",
            "checkpoints"
        )

        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")

        ckpt = get_model_list(ckpt_dir, "net")[-1]

        # ---------- 3. Trainer ----------
        trainer = globals()[config["opt_approach"]](
            net,
            ckpt_dir,
            config
        )

        trainer.resume(config, ckpt, not torch.cuda.is_available())
        trainer.eval()

        # ---------- 4. Eval + ErrorMap ----------
        eval_net(
            trainer.net,                 # ✅ 一定要用 trainer.net
            test_loader,
            save_errormap=True,
            errormap_dir=f"errormaps/{name}_T_{i}"
        )




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config yaml"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="avg_5_models",
        choices=["avg_5_models", "best_eval"],
        help="Evaluation mode"
    )
    args = parser.parse_args()

    if args.mode == "avg_5_models":
        avg_5_models(args.config)
    elif args.mode == "best_eval":
        net, test_loader, _ = load_model(args.config)
        with torch.no_grad():
            eval_net(
                net,
                test_loader,
                save_errormap=True,
                errormap_dir="errormaps/best_eval"
            )




