import os
import shutil
import random

def split_dataset_by_class(root_dir, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42):
    random.seed(seed)

    input_root = os.path.join(root_dir, "input")
    target_root = os.path.join(root_dir, "target")
    output_dirs = ["train", "val", "test"]
    categories = ["c", "l", "w", "h"]

    # 建立 train/val/test 資料夾結構
    for split in output_dirs:
        for subdir in ["input", "target"]:
            for category in categories:
                os.makedirs(os.path.join(root_dir, split, subdir, category), exist_ok=True)

    # 對每個類別各自進行分割
    for category in categories:
        input_dir = os.path.join(input_root, category)
        target_dir = os.path.join(target_root, category)

        input_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        total = len(input_files)

        if total == 0:
            print(f"⚠️ 類別 {category} 沒有圖片，略過。")
            continue

        # 打亂順序
        random.shuffle(input_files)

        # 計算分割點
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))

        splits = {
            "train": input_files[:train_end],
            "val": input_files[train_end:val_end],
            "test": input_files[val_end:]
        }

        # 開始複製
        for split_name, files in splits.items():
            for f in files:
                suffix = f.split("input_")[-1]  # 後綴名稱，如 "1.png"
                target_name = f"target_{suffix}"

                src_input = os.path.join(input_dir, f)
                src_target = os.path.join(target_dir, target_name)

                dst_input = os.path.join(root_dir, split_name, "input", category, f)
                dst_target = os.path.join(root_dir, split_name, "target", category, target_name)

                # 確保 target 存在再複製
                if os.path.exists(src_target):
                    shutil.copy2(src_input, dst_input)
                    shutil.copy2(src_target, dst_target)
                else:
                    print(f"⚠️ 找不到對應 target：{src_target}")

        print(f"✅ {category} 分割完成 — 總數：{total} 份 → "
              f"train: {len(splits['train'])}, val: {len(splits['val'])}, test: {len(splits['test'])}")

    print("🎉 所有類別分割完成！")


if __name__ == "__main__":
    root_dir = f"D:/Users/peggy/Dataset/texture/data"  # ← 改成你的路徑
    split_dataset_by_class(root_dir)
