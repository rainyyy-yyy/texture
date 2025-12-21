import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import argparse
from models_v3 import UNetGenerator

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default="D:\\Pix2Pix\\breast_cancer(SUB_6000)\\Dataset_4")
args = parser.parse_args()

# 2. 設置設備與轉換
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 3. 載入生成器模型
ckpt = torch.load("../TWCC訓練/SUB001_4.pth", map_location=device, weights_only=False)
generator = UNetGenerator().to(device)
generator.load_state_dict(ckpt["generator"])
generator.eval()

# 4. 測試函數
def test_image(input_path, output_path):
    # 讀取16bit灰階圖
    image = Image.open(input_path).convert("I;16")
    # 轉8bit
    input_img_8bit = image.point(lambda i: i * (255.0/65535)).convert("L")
    # 與訓練一致的transform
    input_image = transform(input_img_8bit).unsqueeze(0).to(device)

    with torch.no_grad():
        fake_image = generator(input_image)

    fake_image = fake_image.squeeze().cpu()
    fake_image = (fake_image + 1) / 2.0
    fake_image = (fake_image * 255).clamp(0, 255).byte()
    result = Image.fromarray(fake_image.numpy())
    result.save(output_path)

# 5. 測試所有圖片
test_dir = os.path.join(args.data, "test")
result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

for img_name in os.listdir(test_dir):
    input_path = os.path.join(test_dir, img_name)
    output_path = os.path.join(result_dir, img_name)
    test_image(input_path, output_path)
    print(f"生成完成：{output_path}")

# python pix2pix_generator_3.py