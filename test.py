import os
import glob
import torch
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from models.models_v3 import UNetGenerator
import argparse
from math import log10

# -----------------------
# 參數設定
# -----------------------
parser = argparse.ArgumentParser(description="Texture 模型效能評估（含形態補全與原始比較）")
parser.add_argument('--weights_dir', type=str, default='D:/Users/peggy/Github/output',
                    help='pth 所在資料夾')
parser.add_argument('--test_dirs', type=str, nargs='+',
                    default=['D:/Users/peggy/Dataset/texture/texture/test'],
                    help='test 資料夾 (含 input/target 子資料夾)')
parser.add_argument('--output_dir', type=str, default='D:/Users/peggy/Dataset/texture/output/9',
                    help='儲存生成圖片的資料夾')
parser.add_argument('--threshold', type=int, default=128, help='二值化閾值')
parser.add_argument('--img_size', type=int, default=512)
parser.add_argument('--kernel_size', type=int, default=5, help='形態學補全核大小 (越大補洞越多)')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備: {device}")

# -----------------------
# 安全版 PSNR（避免除以 0）
# -----------------------
def safe_psnr(img1, img2, data_range=255.0):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse < 1e-10:
        return 100.0  # 當 MSE 幾乎為 0 時，回傳極高值而非無限大
    return 10 * log10((data_range ** 2) / mse)

# -----------------------
# 形態學補全函式 (白底黑瑕疵 + 可調整去雜點大小)
# -----------------------
def morphology_fill(
    img, 
    kernel_size=5, 
    threshold=128, 
    min_area_px=None,          # 小於此像素數的瑕疵直接去掉
    max_isolated_area=300,     # 小於此面積才考慮孤立刪除
    max_isolated_dist=100,     # 孤立距離判斷
    debug=False
):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 白底黑瑕疵 → 黑底白瑕疵
    img_inv = 255 - img
    _, binary = cv2.threshold(img_inv, threshold, 255, cv2.THRESH_BINARY)

    # 開運算初步平滑
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 連通區域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
    cleaned = np.zeros_like(opened)
    centroids_list = [tuple(centroids[i]) for i in range(1, num_labels)]

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        c = centroids[i]
        other_centroids = [tuple(centroids[j]) for j in range(1, num_labels) if j != i]
        min_dist = min([np.linalg.norm(np.array(c)-np.array(oc)) for oc in other_centroids], default=0)

        # 判斷順序
        if min_area_px is not None and area < min_area_px:
            if min_dist <= max_isolated_dist:
                cleaned[labels == i] = 255  # 保留
            # else 刪掉 (預設為 0)
        elif area < max_isolated_area:
            if min_dist <= max_isolated_dist:
                cleaned[labels == i] = 255  # 保留
            # else 刪掉 (預設為 0)
        else:
            # area >= max_isolated_area → 保留
            cleaned[labels == i] = 255

    # 閉運算補洞
    closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    # 嘗試連接虛線
    kernel_link = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size+4, kernel_size+1))
    dilated = cv2.dilate(closed, kernel_link, iterations=1)
    linked = cv2.erode(dilated, kernel_link, iterations=1)

    # 平滑化 + 反相回白底黑瑕疵
    result = cv2.medianBlur(linked, 3)
    result_final = 255 - result

    if debug:
        cv2.imshow("binary", binary)
        cv2.imshow("opened", opened)
        cv2.imshow("cleaned", cleaned)
        cv2.imshow("closed", closed)
        cv2.imshow("linked", linked)
        cv2.imshow("result_final", result_final)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result_final

# -----------------------
# 工具函式
# -----------------------
def imread_unicode(path):
    arr = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"讀取失敗: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def load_image(path):
    img = imread_unicode(path)
    img = cv2.resize(img, (args.img_size, args.img_size))
    img = img.astype(np.float32) / 255.0 * 2 - 1
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

def denormalize(tensor):
    return (tensor * 0.5 + 0.5).clamp(0,1)

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
# 讀取權重
# -----------------------
weights_files = sorted(glob.glob(os.path.join(args.weights_dir, "*.pth")))
if not weights_files:
    print("！找不到任何權重檔案")
    exit(0)
print(f"找到 {len(weights_files)} 個權重檔案")

num_weights = len(weights_files)
num_tests = len(args.test_dirs)

if num_weights > num_tests:
    print(f"！權重數量 ({num_weights}) > 測試資料夾數量 ({num_tests})，將循環使用 test_dirs。")
elif num_weights < num_tests:
    print(f"⚠️ 測試資料夾 ({num_tests}) 多於權重 ({num_weights})，多出的資料夾將被忽略。")

test_dirs_expanded = [args.test_dirs[i % num_tests] for i in range(num_weights)]

# -----------------------
# 主程式
# -----------------------
for idx, (weight_path, test_dir) in enumerate(zip(weights_files, test_dirs_expanded)):
    model_name = os.path.splitext(os.path.basename(weight_path))[0]
    print(f"\n=== [{idx+1}/{len(weights_files)}] 評估模型 {model_name} ===")
    print(f"測試資料夾: {test_dir}")

    input_root = os.path.join(test_dir, "input")
    target_root = os.path.join(test_dir, "target")
    if not os.path.isdir(input_root) or not os.path.isdir(target_root):
        print("！缺少 input 或 target 資料夾，跳過。")
        continue

    generator = UNetGenerator(in_channels=1, out_channels=1).to(device)
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    if 'generator' in ckpt:
        generator.load_state_dict(ckpt['generator'])
    elif 'model_state_dict' in ckpt:
        generator.load_state_dict(ckpt['model_state_dict'])
    else:
        generator.load_state_dict(ckpt)
    generator.eval()

    input_files = sorted(
        [os.path.join(root, f)
         for root, _, files in os.walk(input_root)
         for f in files if f.lower().endswith('.png')]
    )
    if not input_files:
        print("！找不到任何輸入圖片。")
        continue

    ssim_list, psnr_list = [], []
    acc_raw, prec_raw, rec_raw = [], [], []
    acc_fill, prec_fill, rec_fill = [], [], []

    out_model_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(out_model_dir, exist_ok=True)

    for input_path in input_files:
        fname = os.path.basename(input_path)
        suffix = fname.split("input_")[-1]
        rel_folder = os.path.relpath(os.path.dirname(input_path), input_root)
        target_path = os.path.join(target_root, rel_folder, f"target_{suffix}")

        if not os.path.exists(target_path):
            continue

        try:
            input_tensor = load_image(input_path)
            target_tensor = load_image(target_path)
            with torch.no_grad():
                fake_tensor = generator(input_tensor)

            input_np = (denormalize(input_tensor).cpu().numpy()[0,0] * 255).astype(np.uint8)
            target_np = (denormalize(target_tensor).cpu().numpy()[0,0] * 255).astype(np.uint8)
            fake_np_raw = (denormalize(fake_tensor).cpu().numpy()[0,0] * 255).astype(np.uint8)

            # --- 形態學補全 ---
            fake_np_fill = morphology_fill(
                                            fake_np_raw, 
                                            kernel_size=5, 
                                            threshold=128, 
                                            min_area_px=50,          # 去掉超小瑕疵
                                            max_isolated_area=300,   # 小於此面積才考慮孤立刪除
                                            max_isolated_dist=70,   # 孤立判斷距離
                                            debug=False
                                        )


            # --- 評估 ---
            ssim_val = ssim(target_np, fake_np_fill, data_range=255)
            psnr_val = safe_psnr(target_np, fake_np_fill, data_range=255)
            ssim_list.append(ssim_val)
            psnr_list.append(psnr_val)

            # 原始 vs 補全
            acc1, prec1, rec1 = calc_metrics(target_np, fake_np_raw, threshold=args.threshold)
            acc2, prec2, rec2 = calc_metrics(target_np, fake_np_fill, threshold=args.threshold)

            acc_raw.append(acc1); prec_raw.append(prec1); rec_raw.append(rec1)
            acc_fill.append(acc2); prec_fill.append(prec2); rec_fill.append(rec2)

            # 顯示這張圖片的評估結果
            print(f"[{fname}]  Acc={acc2:.4f}  Prec={prec2:.4f}  Rec={rec2:.4f}")

            # 儲存結果
            out_folder = os.path.join(out_model_dir, rel_folder)
            os.makedirs(out_folder, exist_ok=True)
            cv2.imwrite(os.path.join(out_folder, f"fake_{suffix}"), fake_np_fill)

        except Exception as e:
            print(f"！無法處理 {fname}: {e}")

    if ssim_list:
        print(f"\n模型 {model_name} 平均指標：")
        print(f"SSIM={np.mean(ssim_list):.4f} | PSNR={np.mean(psnr_list):.2f}")
        print(f"原始 Fake → Acc={np.mean(acc_raw):.4f}, Prec={np.mean(prec_raw):.4f}, Rec={np.mean(rec_raw):.4f}")
        print(f"補全 Fake → Acc={np.mean(acc_fill):.4f}, Prec={np.mean(prec_fill):.4f}, Rec={np.mean(rec_fill):.4f}")
    else:
        print(f"！模型 {model_name} 無有效結果。")
