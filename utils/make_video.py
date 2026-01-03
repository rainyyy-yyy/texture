import os
import cv2
import torch
import numpy as np
from texture.models.models_v3 import UNetGenerator
import argparse

# 指令：python -m texture.utils.make_video

# -----------------------
# 參數設定
# -----------------------
parser = argparse.ArgumentParser(description="從影片生成 Pix2Pix Fake 對比影片 (含形態學補全)")
parser.add_argument('--video_path', type=str, default='D:/Users/peggy/Dataset/texture/test.mp4', help='輸入影片路徑')
parser.add_argument('--weights_path', type=str, default='D:/Users/peggy/Github/output/checkpoint_epoch_70.pth', help='模型權重路徑')
parser.add_argument('--output_dir', type=str, default='D:/Users/peggy/Dataset/texture/video_output', help='輸出資料夾')
parser.add_argument('--frame_interval', type=int, default=10, help='取樣幀距（每多少幀取一次）')
parser.add_argument('--img_size', type=int, default=512, help='輸入圖片大小')
parser.add_argument('--kernel_size', type=int, default=5, help='形態學核大小')
parser.add_argument('--threshold', type=int, default=128, help='二值化閾值')
parser.add_argument('--output_video_name', type=str, default='test.mp4', help='輸出影片名稱')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備: {device}")

# -----------------------
# 模型載入
# -----------------------
generator = UNetGenerator(in_channels=1, out_channels=1).to(device)
ckpt = torch.load(args.weights_path, map_location=device, weights_only=False)
if 'generator' in ckpt:
    generator.load_state_dict(ckpt['generator'])
elif 'model_state_dict' in ckpt:
    generator.load_state_dict(ckpt['model_state_dict'])
else:
    generator.load_state_dict(ckpt)
generator.eval()
print(f"已載入模型權重: {args.weights_path}")

# -----------------------
# 形態學補全函式
# -----------------------
def morphology_fill(
    img,
    kernel_size=5,
    threshold=128,
    min_area_px=50,
    max_isolated_area=300,
    max_isolated_dist=70,
):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_inv = 255 - img
    _, binary = cv2.threshold(img_inv, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
    cleaned = np.zeros_like(opened)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area_px:
            continue
        cleaned[labels == i] = 255

    centroids_list = [tuple(centroids[i]) for i in range(1, num_labels)]
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= max_isolated_area:
            continue
        c = centroids[i]
        min_dist = min(
            [np.linalg.norm(np.array(c)-np.array(other_c)) for j, other_c in enumerate(centroids_list) if j != i-1],
            default=0
        )
        if min_dist > max_isolated_dist:
            cleaned[labels == i] = 0

    closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    kernel_link = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size+4, kernel_size+1))
    dilated = cv2.dilate(closed, kernel_link, iterations=1)
    linked = cv2.erode(dilated, kernel_link, iterations=1)
    result = cv2.medianBlur(linked, 3)
    result_final = 255 - result
    return result_final

# -----------------------
# 工具函式
# -----------------------
def preprocess_frame(frame):
    if frame.ndim == 3:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        frame_gray = frame
    frame_resized = cv2.resize(frame_gray, (args.img_size, args.img_size))
    tensor = torch.from_numpy(frame_resized.astype(np.float32) / 255.0 * 2 - 1)
    return tensor.unsqueeze(0).unsqueeze(0).to(device)

def denormalize(tensor):
    return (tensor * 0.5 + 0.5).clamp(0,1)

# -----------------------
# 讀取影片與建立輸出影片
# -----------------------
cap = cv2.VideoCapture(args.video_path)
if not cap.isOpened():
    print(f"無法開啟影片: {args.video_path}")
    exit(0)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"影片資訊: {frame_count} 幀, FPS={fps:.2f}")

out_path = os.path.join(args.output_dir, args.output_video_name)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None  # 延後建立直到第一幀知道尺寸

frame_idx = 0
save_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % args.frame_interval == 0:
        input_tensor = preprocess_frame(frame)
        with torch.no_grad():
            fake_tensor = generator(input_tensor)

        input_np = (denormalize(input_tensor).cpu().numpy()[0,0] * 255).astype(np.uint8)
        fake_np_raw = (denormalize(fake_tensor).cpu().numpy()[0,0] * 255).astype(np.uint8)

        fake_np_fill = morphology_fill(
            fake_np_raw,
            kernel_size=args.kernel_size,
            threshold=args.threshold,
            min_area_px=50,
            max_isolated_area=300,
            max_isolated_dist=70
        )

        combined = np.hstack((input_np, fake_np_fill))

        # 建立 VideoWriter（在第一幀確定尺寸後）
        if out is None:
            h, w = combined.shape
            out = cv2.VideoWriter(out_path, fourcc, fps / args.frame_interval, (w, h), isColor=False)

        out.write(combined)
        print(f"已處理幀 {save_idx}")
        save_idx += 1

    frame_idx += 1

cap.release()
if out:
    out.release()

print(f"\n🎬 已輸出影片: {out_path}")
print(f"共生成 {save_idx} 幀，每幀大小 {w}x{h}")
