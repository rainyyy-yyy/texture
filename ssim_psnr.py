import os
import glob
import torch
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from models.models_v3 import UNetGenerator
import argparse

# -----------------------
# 參數設定
# -----------------------
parser = argparse.ArgumentParser(description="Texture 模型效能評估 (取 Acc / Prec / Rec 前三名)")
parser.add_argument('--weights_dir', type=str, default='D:/Users/peggy/Github/output/Texture_2025_12_21_16_57',
                    help='pth 所在資料夾')
parser.add_argument('--test_dirs', type=str, nargs='+',
                    default=['D:/Users/peggy/Dataset/texture/texture/test'],
                    help='測試資料夾 (含 input/target 子資料夾)')
parser.add_argument('--threshold', type=int, default=128, help='二值化閾值')
parser.add_argument('--img_size', type=int, default=512)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備: {device}")

# -----------------------
# 工具函式
# -----------------------
def imread_unicode(path):
    arr = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"讀取圖像失敗: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def load_image(path):
    img = imread_unicode(path)
    img = cv2.resize(img, (args.img_size, args.img_size))
    img = img.astype(np.float32) / 255.0 * 2 - 1
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
    return tensor

def denormalize(tensor):
    return (tensor * 0.5 + 0.5).clamp(0, 1)

def calc_metrics(gt, pred, threshold=128):
    gt_bin = (gt >= threshold).astype(np.uint8)
    pred_bin = (pred >= threshold).astype(np.uint8)
    TP = np.sum((gt_bin == 1) & (pred_bin == 1))
    TN = np.sum((gt_bin == 0) & (pred_bin == 0))
    FP = np.sum((gt_bin == 0) & (pred_bin == 1))
    FN = np.sum((gt_bin == 1) & (pred_bin == 0))
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    return accuracy, precision, recall

# -----------------------
# 權重
# -----------------------
weights_files = sorted(glob.glob(os.path.join(args.weights_dir, "*.pth")))
if not weights_files:
    print("找不到任何權重檔案")
    exit(0)
print(f"找到 {len(weights_files)} 個權重檔案")

num_weights = len(weights_files)
num_tests = len(args.test_dirs)
if num_weights > num_tests:
    print(f"權重數量 ({num_weights}) > 測試資料夾數量 ({num_tests})，將循環使用 test_dirs。")
test_dirs_expanded = [args.test_dirs[i % num_tests] for i in range(num_weights)]

# -----------------------
# 主程式
# -----------------------
results = []

for weight_path, test_dir in zip(weights_files, test_dirs_expanded):
    model_name = os.path.splitext(os.path.basename(weight_path))[0]
    print(f"\n=== 評估模型 {model_name} ===")
    print(f"測試資料夾: {test_dir}")

    input_root = os.path.join(test_dir, "input")
    target_root = os.path.join(test_dir, "target")
    if not os.path.isdir(input_root) or not os.path.isdir(target_root):
        print("缺少 input 或 target 資料夾，跳過")
        continue

    # 載入模型
    generator = UNetGenerator(in_channels=1, out_channels=1).to(device)
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    if 'generator' in ckpt:
        generator.load_state_dict(ckpt['generator'])
    elif 'model_state_dict' in ckpt:
        generator.load_state_dict(ckpt['model_state_dict'])
    else:
        generator.load_state_dict(ckpt)
    generator.eval()

    # 收集 input
    input_files = []
    for root, _, files in os.walk(input_root):
        for f in files:
            if f.lower().endswith('.png'):
                input_files.append(os.path.join(root, f))
    input_files.sort()
    if not input_files:
        print("找不到任何輸入圖片，跳過")
        continue

    ssim_list, psnr_list, acc_list, prec_list, rec_list = [], [], [], [], []

    for input_path in input_files:
        fname = os.path.basename(input_path)
        suffix = fname.split("input_")[-1]
        rel_folder = os.path.relpath(os.path.dirname(input_path), input_root)
        target_path = os.path.join(target_root, rel_folder, f"target_{suffix}")

        if not os.path.exists(target_path):
            print(f"找不到對應 target：{target_path}")
            continue

        try:
            input_tensor = load_image(input_path)
            target_tensor = load_image(target_path)
            with torch.no_grad():
                fake_tensor = generator(input_tensor)

            target_np = (denormalize(target_tensor).cpu().numpy()[0, 0] * 255).astype(np.uint8)
            fake_np = (denormalize(fake_tensor).cpu().numpy()[0, 0] * 255).astype(np.uint8)

            ssim_val = ssim(target_np, fake_np, data_range=255)
            psnr_val = psnr(target_np, fake_np, data_range=255)
            acc, prec, rec = calc_metrics(target_np, fake_np, threshold=args.threshold)

            ssim_list.append(ssim_val)
            psnr_list.append(psnr_val)
            acc_list.append(acc)
            prec_list.append(prec)
            rec_list.append(rec)

        except Exception as e:
            print(f"無法處理 {fname}: {e}")

    if ssim_list:
        mean_ssim = np.mean(ssim_list)
        mean_psnr = np.mean(psnr_list)
        mean_acc = np.mean(acc_list)
        mean_prec = np.mean(prec_list)
        mean_rec = np.mean(rec_list)

        print(f"\n模型 {model_name} 平均指標：")
        print(f"SSIM={mean_ssim:.4f} | PSNR={mean_psnr:.2f} | Acc={mean_acc:.4f} | Prec={mean_prec:.4f} | Rec={mean_rec:.4f}")

        results.append({
            "model": model_name,
            "acc": mean_acc,
            "prec": mean_prec,
            "rec": mean_rec
        })
    else:
        print(f"模型 {model_name} 無成功處理的圖片。")

# -----------------------
# 統計前三名
# -----------------------
if results:
    top_acc = sorted(results, key=lambda x: x["acc"], reverse=True)[:3]
    top_prec = sorted(results, key=lambda x: x["prec"], reverse=True)[:3]
    top_rec = sorted(results, key=lambda x: x["rec"], reverse=True)[:3]

    print("\nTop 3 Accuracy Models:")
    for r in top_acc:
        print(f" - {r['model']}: Accuracy={r['acc']:.4f}")

    print("\nTop 3 Precision Models:")
    for r in top_prec:
        print(f" - {r['model']}: Precision={r['prec']:.4f}")

    print("\nTop 3 Recall Models:")
    for r in top_rec:
        print(f" - {r['model']}: Recall={r['rec']:.4f}")
else:
    print("\n沒有任何有效結果。")
