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
parser.add_argument('--output_dir', type=str, default='D:/Users/peggy/Dataset/texture/output/13',
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
    min_area_px=50,
    max_isolated_area=300,
    max_isolated_dist=100,
    debug=False,
    debug_dir=None,
    prefix="debug"
):
    """
    白底黑瑕疵 → 黑底白瑕疵
    以最大瑕疵團邊緣為中心，逐步吸收距離小於 max_isolated_dist 的瑕疵團。
    - 紅點代表被刪除的瑕疵團，標示距離與面積。
    - 綠點代表保留的瑕疵團。
    """

    import os
    import numpy as np
    import cv2

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # === Step 1. 白底黑瑕疵 → 黑底白瑕疵 ===
    img_inv = 255 - img
    _, binary = cv2.threshold(img_inv, threshold, 255, cv2.THRESH_BINARY)

    # === Step 2. 開運算去雜點 ===
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if num_labels <= 1:
        return img  # 沒有瑕疵直接回傳原圖

    h, w = img.shape[:2]
    cleaned = np.zeros_like(opened)

    # === Step 3. 找出最大瑕疵團 ===
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_idx = np.argmax(areas) + 1
    main_mask = np.uint8(labels == max_idx) * 255

    grouped_indices = {max_idx}
    changed = True

    # === Step 4. 以邊緣距離吸收相鄰瑕疵 ===
    while changed:
        changed = False
        contours_main, _ = cv2.findContours(main_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours_main:
            break
        contour_main = max(contours_main, key=cv2.contourArea)

        for i in range(1, num_labels):
            if i in grouped_indices:
                continue
            mask_i = np.uint8(labels == i) * 255
            contours_i, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours_i:
                continue
            contour_i = max(contours_i, key=cv2.contourArea)

            # 計算最短邊緣距離（已修正格式錯誤）
            dist = np.inf
            for p in contour_i:
                x, y = p.ravel()
                d = cv2.pointPolygonTest(contour_main, (float(x), float(y)), True)
                dist = min(dist, abs(d))

            # 若在距離內 → 吸收進大團
            if dist <= max_isolated_dist:
                grouped_indices.add(i)
                main_mask = cv2.bitwise_or(main_mask, mask_i)
                changed = True

    # === Step 5. 依面積與距離條件保留 / 刪除 ===
    keep_flags = []
    contours_main, _ = cv2.findContours(main_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_main:
        contour_main = max(contours_main, key=cv2.contourArea)
    else:
        contour_main = []

    for i in range(1, num_labels):
        mask_i = np.uint8(labels == i) * 255
        area = stats[i, cv2.CC_STAT_AREA]
        c = np.array(centroids[i])

        if i in grouped_indices:
            keep = True
            dist = 0
        else:
            # 計算邊緣距離（安全版）
            dist = np.inf
            contours_i, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours_i and len(contours_main) > 0:
                contour_i = max(contours_i, key=cv2.contourArea)
                for p in contour_i:
                    x, y = p.ravel()
                    d = cv2.pointPolygonTest(contour_main, (float(x), float(y)), True)
                    dist = min(dist, abs(d))

            # 面積與距離判斷
            if area < min_area_px and dist > max_isolated_dist:
                keep = False
            elif area < min_area_px and dist <= max_isolated_dist:
                keep = True
            elif area < max_isolated_area and dist > max_isolated_dist:
                keep = False
            elif area < max_isolated_area and dist <= max_isolated_dist:
                keep = True
            else:
                keep = True

        if keep:
            cleaned[labels == i] = 255
        keep_flags.append((tuple(c), area, dist, keep))

    # === Step 6. 補洞 + 平滑 + 反相 ===
    closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    kernel_link = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size + 4, kernel_size + 1))
    dilated = cv2.dilate(closed, kernel_link, iterations=1)
    linked = cv2.erode(dilated, kernel_link, iterations=1)
    result = cv2.medianBlur(linked, 3)
    result_final = 255 - result

    # === Step 7. Debug 輸出 ===
    if debug and debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        # 1️⃣ 紅綠點圖（紅：刪除，綠：保留）
        vis1 = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        for c, area, dist, keep in keep_flags:
            color = (0, 255, 0) if keep else (0, 0, 255)
            cv2.circle(vis1, (int(c[0]), int(c[1])), 3, color, -1)
            # 列紅點資訊（距離與面積）
            if not keep:
                cv2.putText(vis1, f"{int(dist)}px {int(area)}px",
                            (int(c[0]) + 5, int(c[1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.imwrite(os.path.join(debug_dir, f"{prefix}_red_green.png"), vis1)

        # 2️⃣ 清理後（仍有紅綠點）
        vis2 = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        for c, area, dist, keep in keep_flags:
            color = (0, 255, 0) if keep else (0, 0, 255)
            cv2.circle(vis2, (int(c[0]), int(c[1])), 3, color, -1)
            if not keep:
                cv2.putText(vis2, f"{int(dist)}px {int(area)}px",
                            (int(c[0]) + 5, int(c[1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.imwrite(os.path.join(debug_dir, f"{prefix}_cleaned_marked.png"), vis2)

        # 3️⃣ 最終結果圖（無標記）
        cv2.imwrite(os.path.join(debug_dir, f"{prefix}_final.png"), result_final)

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
            prefix_name = os.path.splitext(fname)[0]
            fake_np_fill = morphology_fill(
                fake_np_raw,
                kernel_size=5,
                threshold=128,
                min_area_px=49,
                max_isolated_area=300,
                max_isolated_dist=250,
                debug=True,
                debug_dir="D:/Users/peggy/Dataset/texture/output/13/debug",
                prefix=prefix_name
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
