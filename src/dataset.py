import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os
import numpy as np

class InpaintingDataset(Dataset):
    def __init__(self, root_dir, mode='train', img_size=512):
        self.root_dir = root_dir
        self.mode = mode
        self.img_size = img_size
        self.files = os.listdir(os.path.join(root_dir, mode))

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.files)

    def load_edge(self,img):
        gray_image = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)
        edges = Image.fromarray(edges)
        return edges


    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.mode, self.files[idx])
        self.img = Image.open(img_path).convert('RGB')

        # 1. Ground Truth Edge
        edge = self.load_edge(self.img)

        # Transform
        img_t = self.transform(self.img)
        edge_t = transforms.ToTensor()(transforms.Resize((self.img_size, self.img_size))(edge))

        # 2. Mask
        mask = self.load_mask(128,128,64,64)

        return img_t, edge_t, mask

    def load_mask(self, x, y, mask_h, mask_w):
        h,w = self.img.size

        mask = np.zeros((h, w), np.float32)

        mask[y:y+mask_h, x:x+mask_w] = 1.0
        return torch.from_numpy(mask).unsqueeze(0).float()


if __name__ == "__main__":
    dataset = InpaintingDataset(root_dir='./datasets/img', mode='train', img_size=512)
    print("Dataset length:", len(dataset))
    img, edge, mask = dataset[0]
    print("Image shape:", img.shape)
    print("Edge shape:", edge.shape)
    print("Mask shape:", mask.shape)