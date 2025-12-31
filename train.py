import os
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timedelta
import torchvision.transforms as transforms
from conf import settings as Settings
from models.models_v3 import UNetGenerator, PatchGANDiscriminator
from models.pytorch_msssim import ms_ssim
from torchvision.utils import make_grid
from torchvision.utils import save_image
from utils.texture_dataloader import get_dataloader
from skimage.metrics import structural_similarity as ssim

# python 常用工具/train_pix2pix_3_TWCC實驗.py --data breast_cancer(SUB_6000)/Dataset_0 --adv_use 1 --ssim_use 0 --l1_use 0 --type SUB
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=Settings.BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=Settings.EPOCHS)
    parser.add_argument('--img', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data', type=str, default=Settings.DATASET_ROOT)
    parser.add_argument('--resume', type=str, default=None, help='checkpoint 路徑，若要 resume 請指定')
    parser.add_argument('--adv_use', type=int, default=0, help='是否使用對抗損失')
    parser.add_argument('--ssim_use', type=int, default=0, help='是否使用SSIM損失')
    parser.add_argument('--l1_use', type=int, default=1, help='是否使用L1損失')
    parser.add_argument('--g_lr', type=float, default=Settings.G_LR)
    parser.add_argument('--d_lr', type=float, default=Settings.D_LR)
    parser.add_argument('--type', type=str, default='SUB')
    return parser.parse_args()

# ============================
# 3. Training Function
# ============================
def train(dataloader, generator, discriminator, g_optim, d_optim, l1_loss, bce_loss, device, a, b, c):

    def to01(x):
        # 若你的模型輸出/標籤是 [-1,1]，先轉回 [0,1] 再算 SSIM/PSNR
        return (x + 1) / 2

    generator.train()
    discriminator.train()

    for i, batch in enumerate(dataloader):
        input_image = batch["edge_images"].to(device)
        target_image = batch["color_images"].to(device)

        # Train Discriminator
        fake_image = generator(input_image)
        d_real = discriminator(input_image, target_image)
        d_fake = discriminator(input_image, fake_image.detach())
        d_loss_real = bce_loss(d_real, torch.ones_like(d_real))
        d_loss_fake = bce_loss(d_fake, torch.zeros_like(d_fake))
        d_loss = (d_loss_real + d_loss_fake) / 2

        d_optim.zero_grad()
        d_loss.backward()
        d_optim.step()

        g_adv_loss = 0
        g_l1_loss = 0
        g_ssim_loss = 0

        # Train Generator
        if a != 0:
            d_fake = discriminator(input_image, fake_image)
            g_adv_loss = bce_loss(d_fake, torch.ones_like(d_fake)) * 4.5
        if b != 0:
            g_l1_loss = l1_loss(fake_image, target_image) * 40

        # MS-SSIM（建議優於單尺度 SSIM），作為 loss 使用 1 - MS-SSIM
        if c != 0:
            fake01 = to01(fake_image).clamp(0, 1)
            tgt01  = to01(target_image).clamp(0, 1)
            g_ssim_loss = (1.0 - ms_ssim(fake01, tgt01, size_average=True)) * 16

        g_loss =a * g_adv_loss + b * g_l1_loss + c * g_ssim_loss

        g_optim.zero_grad()
        g_loss.backward()
        g_optim.step()

        if i % 100 == 0:
            print(f"Batch {i}/{len(dataloader)}: D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

# ============================
# 4. validing Function
# ============================
def validate(dataloader, generator, l1_loss, device, out_dir, best_ssim):
    
    # PSNR 用 pixel_max=2.0（[-1,1] 的峰值差）
    def psnr(fake, target):
        mse = np.mean((fake - target) ** 2)
        if mse == 0: return 100.0
        pixel_max = 2.0
        return 20 * np.log10(pixel_max / np.sqrt(mse))

    # 把 [-1, 1] 還原成 [0, 1]，確保無論如何都不會超亮或超暗
    def denormalize(tensor):
        return (tensor * 0.5 + 0.5).clamp(0, 1)

    generator.eval()
    total_ssim = 0
    total_mae = 0
    total_psnr = 0  # 新增 PSNR 計算
    total_samples = 0

    current_best_batch_ssim = -1.0  # 紀錄本次 epoch 最高 batch SSIM
    best_batch_images = None        # 對應 batch 的三種圖 (input, target, fake)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_image = batch["edge_images"].to(device)
            target_image = batch["color_images"].to(device)
            fake_image = generator(input_image)

            # SSIM & MAE (轉 numpy)
            fake_np = fake_image.cpu().numpy()
            target_np = target_image.cpu().numpy()
            batch_ssim = 0
            for j in range(fake_np.shape[0]):
                ssim_val = ssim(fake_np[j][0], target_np[j][0], data_range=2.0)
                total_ssim += ssim_val
                batch_ssim += ssim_val
                mae_val = np.mean(np.abs(fake_np[j][0] - target_np[j][0]))
                total_mae += mae_val
                psnr_val = psnr(fake_np[j][0], target_np[j][0])
                total_psnr += psnr_val
            
            batch_ssim /= fake_np.shape[0]  # 計算當前 batch 平均 SSIM

            # 若此 batch SSIM 高於目前 epoch 最高 → 暫存該 batch
            if batch_ssim > current_best_batch_ssim:
                current_best_batch_ssim = batch_ssim
                best_batch_images = (input_image.clone(), target_image.clone(), fake_image.clone())

            total_samples += fake_np.shape[0]

    avg_ssim = total_ssim / total_samples
    avg_mae = total_mae / total_samples
    avg_psnr = total_psnr / total_samples  # 計算平均 PSNR
    print(f"Validation SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.4f}")
    
    # 若整體 SSIM 優於歷史最佳 → 儲存該 batch 模型
    if avg_ssim > best_ssim and best_batch_images is not None:
        input_image, target_image, fake_image = best_batch_images

        print(f"目前最佳 SSIM: {avg_ssim:.4f}，儲存模型中...")

        # 儲存模型
        model_path = os.path.join(out_dir, "best_model.pth")
        torch.save(generator.state_dict(), model_path)
        print(f"已儲存最佳模型: {model_path}")

        best_ssim = avg_ssim

    return avg_ssim, avg_mae, avg_psnr, best_ssim

# ============================
# 5. Main Training Loop
# ============================
def main(out_root):
    import random
    # TODO: 自訂超參數:
    start_epoch = 0
    epoch_times = []
    patience = 30  # 早停容忍次數，可自行調整
    no_improve_count = 0
    best_objective = float('inf')  # 新增這行
    best_model_path = None

    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    batch_size = args.batch
    num_epochs = args.epochs
    img_size = args.img
    data_root = args.data
    resume_path = args.resume
    g_lr = args.g_lr
    d_lr = args.d_lr

    best_ssim = -1.0

    total = args.adv_use + args.ssim_use + args.l1_use
    adv_use = 3 / total * args.adv_use
    ssim_use = 3 / total * args.ssim_use
    l1_use =3 / total * args.l1_use

    # 建立資料夾
    out_dir = out_root
    os.makedirs(out_dir, exist_ok=True)

    # 設定隨機種子，resume 時也要恢復
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 使用 M60Dataset / get_dataloader
    dataloaders = get_dataloader(
        dataset_name="texture",
        batch_size=batch_size,
        data_root=data_root,
        train_num_workers=4,
        transforms=transform,
        val_num_workers=2,
        test_num_workers=2
    )

    train_loader = dataloaders.train_dataloader
    val_loader = dataloaders.val_dataloader

    generator = UNetGenerator().to(device)
    discriminator = PatchGANDiscriminator().to(device)

    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 張 GPU 進行訓練")
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)

    g_optim = optim.Adam(generator.parameters(), lr=g_lr, betas=(0.5, 0.999))
    d_optim = optim.Adam(discriminator.parameters(), lr=d_lr, betas=(0.5, 0.999))

    l1_loss = nn.L1Loss()
    bce_loss = nn.BCEWithLogitsLoss()

    # resume 機制
    if resume_path is not None:
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"找不到 checkpoint: {resume_path}")
        print(f"載入 checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        # 檢查模型架構
        def check_keys(model, state_dict):
            model_keys = set(model.state_dict().keys())
            ckpt_keys = set(state_dict.keys())
            if model_keys != ckpt_keys:
                diff1 = model_keys - ckpt_keys
                diff2 = ckpt_keys - model_keys
                raise RuntimeError(f"模型架構不符! model多: {diff1}, ckpt多: {diff2}")
        # DataParallel 處理
        gen_state = checkpoint['generator']
        dis_state = checkpoint['discriminator']
        if isinstance(generator, nn.DataParallel):
            check_keys(generator.module, gen_state)
            generator.module.load_state_dict(gen_state)
            check_keys(discriminator.module, dis_state)
            discriminator.module.load_state_dict(dis_state)
        else:
            check_keys(generator, gen_state)
            generator.load_state_dict(gen_state)
            check_keys(discriminator, dis_state)
            discriminator.load_state_dict(dis_state)
        g_optim.load_state_dict(checkpoint['g_optim'])
        d_optim.load_state_dict(checkpoint['d_optim'])
        start_epoch = checkpoint['epoch'] + 1
        epoch_times = checkpoint.get('epoch_times', [])
        best_objective = checkpoint.get('best_val_loss', float('inf'))  # 用 objective_score 取代
        # 恢復隨機種子
        if 'random_state' in checkpoint:
                torch_state = checkpoint['random_state']['torch']
                if not isinstance(torch_state, torch.ByteTensor):
                        torch_state = torch_state.cpu()
                        torch_state = torch.ByteTensor(torch_state)
                torch.set_rng_state(torch_state)
                np.random.set_state(checkpoint['random_state']['numpy'])
                random.setstate(checkpoint['random_state']['python'])
        print(f"Resume 成功，從 epoch {start_epoch} 繼續訓練")

    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}] 開始")
        start_time = time.time()
        
        train(train_loader, generator, discriminator, g_optim, d_optim, l1_loss, bce_loss, device, adv_use, l1_use, ssim_use)
        avg_ssim, avg_mae, avg_psnr, best_ssim = validate(val_loader, generator, l1_loss, device, out_dir, best_ssim)

        Pmax = 30.0
        psnr_bad = max(0.0, 1.0 - (avg_psnr / Pmax))  # 0~1
        objective_score = (0.7 * (1 - avg_ssim)) + (0.1 * avg_mae) + (0.2 * psnr_bad)

        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        avg_time = sum(epoch_times) / len(epoch_times)
        remaining_time_sec = avg_time * (num_epochs - epoch - 1)
        remaining_td = timedelta(seconds=int(remaining_time_sec))
        finish_time = datetime.now() + remaining_td
        print(f"Epoch {epoch+1} 實際花費時間: {int(epoch_time)} 秒")
        print(f"預計剩餘時間: {str(remaining_td)}，預計完成時間: {finish_time.strftime('%H:%M:%S')}")

        # 每 10 個 epoch 儲存一次
        if (epoch + 1) % 10 == 0:
            random_state = {
                'torch': torch.get_rng_state(),
                'numpy': np.random.get_state(),
                'python': random.getstate()
            }
            save_dict = {
                'generator': generator.module.state_dict() if isinstance(generator, nn.DataParallel) else generator.state_dict(),
                'discriminator': discriminator.module.state_dict() if isinstance(discriminator, nn.DataParallel) else discriminator.state_dict(),
                'g_optim': g_optim.state_dict(),
                'd_optim': d_optim.state_dict(),
                'epoch': epoch,
                'epoch_times': epoch_times,
                'best_val_loss': min(objective_score, best_objective),  # 用 objective_score
                'random_state': random_state
            }
            ckpt_path = os.path.join(out_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(save_dict, ckpt_path)
            print(f"已儲存 checkpoint: {ckpt_path}")

        # 儲存最佳模型
        if objective_score < best_objective:  # 用 objective_score 判斷
            best_objective = objective_score
            no_improve_count = 0
            random_state = {
                'torch': torch.get_rng_state(),
                'numpy': np.random.get_state(),
                'python': random.getstate()
            }
            save_dict = {
                'generator': generator.module.state_dict() if isinstance(generator, nn.DataParallel) else generator.state_dict(),
                'discriminator': discriminator.module.state_dict() if isinstance(discriminator, nn.DataParallel) else discriminator.state_dict(),
                'g_optim': g_optim.state_dict(),
                'd_optim': d_optim.state_dict(),
                'epoch': epoch,
                'epoch_times': epoch_times,
                'best_val_loss': best_objective,  # 用 objective_score
                'random_state': random_state
            }
            best_model_path = os.path.join(out_dir, "best_model.pth")
            torch.save(save_dict, best_model_path)
            print(f"*** 已儲存最佳模型: {best_model_path} (objective_score={best_objective:.6f})")

        else:
            no_improve_count += 1
            print(f"驗證 loss 未改善，early stopping 計數: {no_improve_count}/{patience}")

        # early stopping 判斷
        if no_improve_count >= patience:
            print(f"驗證 loss 已連續 {patience} 次未改善，提前停止訓練。最佳模型已儲存於 {best_model_path}")
            break

if __name__ == "__main__":
    # TODO: --device 可能需要改成 0
    # 建立資料夾
    output_file = time.strftime(
            "{}_{}_{}_{}_{}_{}".format("Texture",
                                        time.localtime().tm_year,
                                        time.localtime().tm_mon,
                                        time.localtime().tm_mday,
                                        time.localtime().tm_hour,
                                        time.localtime().tm_min),
        time.localtime()
    )
    out_dir = os.path.join(Settings.OUTPUT_ROOT, output_file)
    os.makedirs(out_dir, exist_ok=True)
    print("=== Start training ===")
    main(out_dir) 