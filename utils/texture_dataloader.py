import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class TextureDataset(Dataset):
    """
    適用於 input_xxx / target_xxx 命名規則、且包含多層子資料夾的 Pix2Pix 資料集。
    結構：
        root/
          input/
            c270/
              input_2_270.png
            c90/
              ...
          target/
            c270/
              target_2_270.png
            c90/
              ...
    """

    def __init__(self, data_root, mode="train", transforms=None):
        self.mode = mode
        self.input_root = os.path.join(data_root, "input")
        self.target_root = os.path.join(data_root, "target")
        self.transforms = transforms

        # 收集所有 input 檔案完整路徑
        self.input_paths = []
        for root, _, files in os.walk(self.input_root):
            for f in files:
                if f.lower().endswith(".png") and f.startswith("input_"):
                    self.input_paths.append(os.path.join(root, f))

        self.input_paths.sort()

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        input_path = self.input_paths[idx]

        # 解析出相對路徑與對應 target 檔名
        rel_path = os.path.relpath(input_path, self.input_root)  # e.g. c270/input_2_270.png
        folder = os.path.dirname(rel_path)                       # e.g. c270
        input_name = os.path.basename(input_path)                # e.g. input_2_270.png

        # 對應 target 檔名
        target_name = input_name.replace("input_", "target_")
        target_path = os.path.join(self.target_root, folder, target_name)

        # 確保 target 存在
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Cannot find target for {input_name} → {target_path}")

        # 讀取灰階影像
        input_image = Image.open(input_path).convert("L")
        target_image = Image.open(target_path).convert("L")

        # 確保大小一致
        if input_image.size != target_image.size:
            target_image = target_image.resize(input_image.size)

        if self.transforms:
            input_image = self.transforms(input_image)
            target_image = self.transforms(target_image)

        return {
            "edge_images": input_image,
            "color_images": target_image,
            "name": os.path.join(folder, input_name)
        }


class SeparatedFolderDataLoader:
    def __init__(self, root, batch_size, train_num_workers, val_num_workers, test_num_workers, transforms):
        train_set = TextureDataset(os.path.join(root, "train"), mode="train", transforms=transforms)
        val_set = TextureDataset(os.path.join(root, "val"), mode="val", transforms=transforms)
        test_set = TextureDataset(os.path.join(root, "test"), mode="test", transforms=transforms)

        self.train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=train_num_workers)
        self.val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=val_num_workers)
        self.test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=test_num_workers)


def get_dataloader(dataset_name, batch_size, data_root, train_num_workers, val_num_workers, test_num_workers, transforms):
    if dataset_name == "texture":
        train_dataset = TextureDataset(os.path.join(data_root, "train"), transforms=transforms)
        val_dataset = TextureDataset(os.path.join(data_root, "val"), transforms=transforms)
        test_dataset = TextureDataset(os.path.join(data_root, "test"), transforms=transforms)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=train_num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=val_num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=test_num_workers)

        return type("DataloaderWrapper", (), {
            "train_dataloader": train_loader,
            "val_dataloader": val_loader,
            "test_dataloader": test_loader
        })()
    else:
        raise KeyError(f"Dataset name error: {dataset_name}")
