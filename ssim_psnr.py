import os
import glob
import torch
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from models.models_v3 import UNetGenerator
import argparse
import csv

# -----------------------
# 參數
# -----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--weights_dir', type=str, default='D:/Users/peggy/Github/output/Pix2Pix_2025_10_28_0_55/Dataset_0')
parser.add_argument('--test_dirs', type=str, nargs='+', 
                    default=['D:/Users/peggy/Dataset/ddsp/Dataset_0/test'])
parser.add_argument('--output_csv', type=str, default='metrics_results.csv')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備: {device}")

# -----------------------
# 讀取權重
# -----------------------
weights_files = sorted([f for f in glob.glob(os.path.join(args.weights_dir, "*.pth")) if os.path.isfile(f)])
if not weights_files:
    print("找不到任何權重檔案")
    exit(0)
print(f"找到 {len(weights_files)} 個權重檔案")

# -----------------------
# 參數設定
# -----------------------
img_size = 256

# -----------------------
# 函數定義
# -----------------------
def imread_unicode(path):
    arr = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"讀取圖像失敗: {path}")
    return img

def load_2input_1target(path):
    img = imread_unicode(path)
    img = cv2.resize(img, (img_size * 3, img_size), interpolation=cv2.INTER_CUBIC)
    img_np = img.astype(np.float32)

    w = img_np.shape[1]
    part_w = w // 3
    input_A = img_np[:, :part_w]
    input_B = img_np[:, part_w:part_w*2]
    target_C = img_np[:, part_w*2:]

    input_A = torch.from_numpy(input_A / 65535.0 * 2 - 1).unsqueeze(0)
    input_B = torch.from_numpy(input_B / 65535.0 * 2 - 1).unsqueeze(0)
    input_tensor = torch.cat([input_A, input_B], dim=0).unsqueeze(0).to(device)

    target_tensor = torch.from_numpy(target_C / 65535.0 * 2 - 1).unsqueeze(0).unsqueeze(0).to(device)

    return input_tensor, target_tensor

def generate_image(generator, input_tensor):
    with torch.no_grad():
        return generator(input_tensor)

def denormalize(tensor):
    return (tensor * 0.5 + 0.5).clamp(0,1)

def compute_confusion_metrics(target_np, pred_np, threshold=128):
    """
    將目標與生成圖二值化後計算 Accuracy, Precision, Recall
    target_np, pred_np: 0~255 uint8
    """
    target_bin = (target_np >= threshold).astype(np.uint8)
    pred_bin = (pred_np >= threshold).astype(np.uint8)

    TP = np.sum((pred_bin==1) & (target_bin==1))
    TN = np.sum((pred_bin==0) & (target_bin==0))
    FP = np.sum((pred_bin==1) & (target_bin==0))
    FN = np.sum((pred_bin==0) & (target_bin==1))

    acc = (TP + TN) / max(TP + TN + FP + FN, 1)
    prec = TP / max(TP + FP, 1)
    rec = TP / max(TP + FN, 1)

    return acc, prec, rec

# -----------------------
# 主程式
# -----------------------
model_results = []

if len(weights_files) != len(args.test_dirs):
    print("！權重數量與 test 資料夾數量不一致，將以最短數量為準")

for weight_path, test_dir in zip(weights_files, args.test_dirs):
    if not os.path.exists(test_dir):
        print(f"！測試資料夾不存在: {test_dir}, 跳過")
        continue

    model_name = os.path.splitext(os.path.basename(weight_path))[0]
    print(f"\n評估模型 {model_name}，測試資料夾: {test_dir}")

    # 載入模型
    generator = UNetGenerator(in_channels=2, out_channels=1).to(device)
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    if 'generator' in ckpt:
        generator.load_state_dict(ckpt['generator'])
    elif 'model_state_dict' in ckpt:
        generator.load_state_dict(ckpt['model_state_dict'])
    else:
        generator.load_state_dict(ckpt)
    generator.eval()

    # 遞迴抓取 PNG
    test_files = sorted(glob.glob(os.path.join(test_dir, "**", "*.png"), recursive=True))
    if not test_files:
        print(f"！沒有找到 PNG 測試圖像，跳過此資料夾")
        continue

    psnr_list, ssim_list = [], []
    acc_list, prec_list, rec_list = [], [], []

    for path in test_files:
        try:
            input_tensor, target_tensor = load_2input_1target(path)
            fake_tensor = generate_image(generator, input_tensor)

            fake_np = denormalize(fake_tensor).squeeze().cpu().numpy()
            target_np = denormalize(target_tensor).squeeze().cpu().numpy()

            psnr_val = psnr(target_np, fake_np, data_range=1.0)
            ssim_val = ssim(target_np, fake_np, data_range=1.0)
            acc, prec, rec = compute_confusion_metrics((target_np*255).astype(np.uint8),
                                                       (fake_np*255).astype(np.uint8))

            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
            acc_list.append(acc)
            prec_list.append(prec)
            rec_list.append(rec)

        except Exception as e:
            print(f"！無法處理 {path}: {e}")
            continue

    if psnr_list:
        avg_psnr = sum(psnr_list)/len(psnr_list)
        avg_ssim = sum(ssim_list)/len(ssim_list)
        avg_acc = sum(acc_list)/len(acc_list)
        avg_prec = sum(prec_list)/len(prec_list)
        avg_rec = sum(rec_list)/len(rec_list)

        print(f"平均 PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, Accuracy: {avg_acc:.4f}, Precision: {avg_prec:.4f}, Recall: {avg_rec:.4f}")

        model_results.append({
            'model': model_name,
            'weight_path': weight_path,
            'psnr': avg_psnr,
            'ssim': avg_ssim,
            'accuracy': avg_acc,
            'precision': avg_prec,
            'recall': avg_rec
        })
    else:
        print(f"！模型 {model_name} 沒有成功評估的圖片。")

# -----------------------
# 輸出 CSV
# -----------------------
if model_results:
    with open(args.output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['model','weight_path','psnr','ssim','accuracy','precision','recall'])
        writer.writeheader()
        for r in model_results:
            writer.writerow(r)
    print(f"\n已儲存結果至 CSV: {args.output_csv}")
else:
    print("！沒有可評估的模型")
