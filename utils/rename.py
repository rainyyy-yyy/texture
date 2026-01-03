import os

def rename_images_recursive(root_dir):
    """
    遍歷 root_dir 下的所有子資料夾，將圖片重新命名為：
    原始檔名 + 最後一層資料夾名稱 + 副檔名
    """
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    for foldername, subfolders, filenames in os.walk(root_dir):
        # 取得目前資料夾名稱（最後一層）
        folder_label = os.path.basename(foldername)
        
        for filename in filenames:
            if filename.lower().endswith(exts):
                old_path = os.path.join(foldername, filename)
                name, ext = os.path.splitext(filename)

                # 檢查是否已經加過資料夾名稱
                if not name.endswith(f"_{folder_label}"):
                    new_name = f"{name}_{folder_label}{ext}"
                    new_path = os.path.join(foldername, new_name)

                    try:
                        os.rename(old_path, new_path)
                        print(f"{old_path} → {new_name}")
                    except Exception as e:
                        print(f"無法重新命名 {old_path}: {e}")

if __name__ == "__main__":
    root_dir = r"D:/Users/peggy/Dataset/texture/target/targett"
    rename_images_recursive(root_dir)
    print("所有圖片重新命名完成")
