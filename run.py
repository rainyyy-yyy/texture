import os
import glob
import torch
import numpy as np
import cv2
from models.models_v3 import UNetGenerator
from skimage.metrics import structural_similarity as ssim
import argparse

# -----------------------
# 參數設定
# -----------------------
torch.serialization.add_safe_globals([np.core.multiarray.scalar])

parser = argparse.ArgumentParser(description="Texture 生成並評估瑕疵影像")
parser.add_argument('--weights_dir', type=str, default='D:/Users/peggy/Github/best pth/test3', help='pth所在資料夾')
parser.add_argument('--test_dirs', type=str, nargs='+',
                    default=['D:/Users/peggy/Dataset/test/13-1/dp'], help='input資料夾')
parser.add_argument('--output_dir', type=str, default='D:/Users/peggy/Dataset/output/13', help='輸出資料夾')
parser.add_argument('--threshold', type=int, default=128, help='二值化閾值，用於瑕疵檢測')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備: {device}")

# -----------------------
# 函數定義
# -----------------------
def imread_unicode(path):
    arr = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"！讀取圖像失敗: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def load_1input_1target(path):
    img = imread_unicode(path)
    h, w = img.shape
    part_w = w // 2
    input_img = img[:, :part_w]
    target_img = img[:, part_w:]

    min_h = min(input_img.shape[0], target_img.shape[0])
    min_w = min(input_img.shape[1], target_img.shape[1])
    input_img = input_img[:min_h, :min_w]
    target_img = target_img[:min_h, :min_w]

    # normalize [-1,1]
    input_img = input_img.astype(np.float32) / 65535.0 * 2 - 1
    target_img = target_img.astype(np.float32) / 65535.0 * 2 - 1

    input_tensor = torch.from_numpy(input_img).unsqueeze(0).unsqueeze(0).to(device)
    target_tensor = torch.from_numpy(target_img).unsqueeze(0).unsqueeze(0).to(device)
    return input_tensor, target_tensor

def denormalize(tensor):
    return (tensor * 0.5 + 0.5).clamp(0,1)

def generate_image(generator, input_tensor):
    with torch.no_grad():
        return generator(input_tensor)

def calc_metrics(gt, pred, threshold=128):
    # 二值化
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
# 主程式
# -----------------------
weights_files = sorted([f for f in glob.glob(os.path.join(args.weights_dir, "*.pth")) if os.path.isfile(f)])
if not weights_files:
    print("！找不到任何權重檔案")
    exit(0)
print(f"找到 {len(weights_files)} 個權重檔案")

for weight_path, test_dir in zip(weights_files, args.test_dirs):
    if not os.path.exists(test_dir):
        print(f"！測試資料夾不存在: {test_dir}, 跳過")
        continue

    model_name = os.path.splitext(os.path.basename(weight_path))[0]
    print(f"\n使用模型 {model_name} 處理資料夾: {test_dir}")

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

    # 輸出資料夾
    out_input_dir = os.path.join(args.output_dir, model_name, 'input')
    out_target_dir = os.path.join(args.output_dir, model_name, 'target')
    out_fake_dir = os.path.join(args.output_dir, model_name, 'fake')
    os.makedirs(out_input_dir, exist_ok=True)
    os.makedirs(out_target_dir, exist_ok=True)
    os.makedirs(out_fake_dir, exist_ok=True)

    # 遞迴抓 PNG
    test_files = []
    for root, dirs, files in os.walk(test_dir):
        for f in files:
            if f.lower().endswith('.png'):
                test_files.append(os.path.join(root, f))
    test_files.sort()

    ssim_list = []
    acc_list, prec_list, rec_list = [], [], []

    for path in test_files:
        fname = os.path.basename(path)
        try:
            input_tensor, target_tensor = load_1input_1target(path)

            _, _, h, w = input_tensor.shape
            orig_h, orig_w = target_tensor.shape[2], target_tensor.shape[3]

            # pad 到 64 整除
            pad_h = ((h+63)//64)*64 - h
            pad_w = ((w+63)//64)*64 - w
            input_padded = torch.nn.functional.pad(input_tensor, (0,pad_w,0,pad_h), mode='reflect')

            # 生成 fake
            fake_tensor = generate_image(generator, input_padded)
            fake_tensor = fake_tensor[:, :, :orig_h, :orig_w]

            # 轉 numpy
            input_np = (denormalize(input_tensor).cpu().numpy()[0,0]*255).astype(np.uint8)
            target_np = (denormalize(target_tensor).cpu().numpy()[0,0]*255).astype(np.uint8)
            fake_np = (denormalize(fake_tensor).cpu().numpy()[0,0]*255).astype(np.uint8)

            # SSIM
            ssim_val = ssim(target_np, fake_np, data_range=255)
            ssim_list.append(ssim_val)

            # Accuracy / Precision / Recall
            acc, prec, rec = calc_metrics(target_np, fake_np, threshold=args.threshold)
            acc_list.append(acc)
            prec_list.append(prec)
            rec_list.append(rec)

            # 儲存圖片
            cv2.imwrite(os.path.join(out_input_dir, fname), input_np)
            cv2.imwrite(os.path.join(out_target_dir, fname), target_np)
            cv2.imwrite(os.path.join(out_fake_dir, fname), fake_np)

            print(f"生成完成: {fname} | SSIM: {ssim_val:.4f} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")

        except Exception as e:
            print(f"！無法處理 {fname}: {e}")

    # 平均指標
    if ssim_list:
        print(f"\n模型 {model_name} 平均 SSIM: {np.mean(ssim_list):.4f}, "
              f"Accuracy: {np.mean(acc_list):.4f}, Precision: {np.mean(prec_list):.4f}, Recall: {np.mean(rec_list):.4f}")
