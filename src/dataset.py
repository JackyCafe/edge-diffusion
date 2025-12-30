import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class InpaintingDataset(Dataset):
    def __init__(self, root_dir, mode='train', img_size=512):
        self.root_dir = root_dir
        self.mode = mode
        self.img_size = img_size
        self.path = os.path.join(root_dir, mode)

        # --- 路徑檢查 ---
        if not os.path.exists(self.path):
            # 嘗試建立資料夾以提示使用者
            try:
                os.makedirs(self.path)
                print(f"提示: 已自動建立資料夾 {self.path}，請放入圖片！")
            except:
                pass
            raise FileNotFoundError(f"找不到資料夾: {self.path}")

        self.files = [f for f in os.listdir(self.path)
                      if f.lower().endswith(('.jpg', '.png', '.jpeg','.pgm'))]

        if len(self.files) == 0:
             raise RuntimeError(f"錯誤：在 {self.path} 內找不到任何圖片 (jpg/png)！")

        print(f"[{mode}] 資料載入成功: {len(self.files)} 張圖片")

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # RGB [-1, 1]
        ])

    def __len__(self):
        return len(self.files)

    def load_edge(self, img):
        # RGB -> Gray -> Canny Edge Detection
        img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(img_gray, 100, 200)
        edges = Image.fromarray(edges)
        return edges

    def load_mask(self, height, width,h_hole=64,w_hole=64):
        mask = np.zeros((height, width), dtype=np.uint8)
        # 隨機矩形遮罩
        # h_hole = np.random.randint(height // 4, height // 2)
        # w_hole = np.random.randint(width // 4, width // 2)
        y = 320
        x = 320
        mask[y:y+h_hole, x:x+w_hole] = 1
        return torch.from_numpy(mask).unsqueeze(0).float() # [1, H, W]

    def __getitem__(self, index):
        img_path = os.path.join(self.path, self.files[index])
        try:
            img = Image.open(img_path).convert('RGB')
            # 1. 產生 Ground Truth Edge
            edge = self.load_edge(img)

            # 2. 轉換與正規化
            img_t = self.transform(img)
            edge_t = transforms.ToTensor()(transforms.Resize((self.img_size, self.img_size))(edge)) # Edge [0, 1]

            # 3. 產生 Mask
            mask = self.load_mask(self.img_size, self.img_size)

            return img_t, edge_t, mask
        except Exception as e:
            print(f"讀取錯誤 {img_path}: {e}")
            # 遇到壞圖隨機換一張
            return self.__getitem__(np.random.randint(0, len(self.files)))