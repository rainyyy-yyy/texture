import torch
import torch.nn.functional as F
from torch import nn
import math

# -----------------------------
# 高斯濾波器生成
# -----------------------------
def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2 / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

# -----------------------------
# 單尺度 SSIM
# -----------------------------
def ssim(img1, img2, window_size=11, window=None, size_average=True, val_range=None):
    L = val_range if val_range is not None else 1.0  # 資料範圍

    padd = window_size // 2
    (_, channel, _, _) = img1.size()
    if window is None:
        window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

# -----------------------------
# 多尺度 MS-SSIM
# -----------------------------
def ms_ssim(img1, img2, window_size=11, size_average=True, val_range=None, weights=None):
    if weights is None:
        weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(img1.device)

    levels = weights.size(0)
    mssim = []
    mcs = []
    for _ in range(levels):
        ssim_map = ssim(img1, img2, window_size=window_size, size_average=False, val_range=val_range)
        cs_map = ssim_map  # for MS-SSIM, we reuse ssim_map for simplicity
        mssim.append(ssim_map.mean())
        mcs.append(cs_map.mean())

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # MS-SSIM 組合公式
    ms_ssim_val = torch.prod((mcs[:-1] ** weights[:-1]) * (mssim[-1] ** weights[-1]))
    return ms_ssim_val if size_average else ms_ssim_val.mean()

# -----------------------------
# 測試
# -----------------------------
if __name__ == "__main__":
    x = torch.rand(1, 1, 256, 256)
    y = torch.rand(1, 1, 256, 256)
    print("MS-SSIM:", ms_ssim(x, y, val_range=1.0))
