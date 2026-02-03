import cv2
import numpy as np
from PIL import Image

def check_image_channels(img_path):
    # --- 方法 A: 使用 OpenCV 檢查物理結構 ---
    # cv2.IMREAD_UNCHANGED 可以保留原始通道（包含 Alpha 通道）
    img_cv = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    if img_cv is None:
        return "無法讀取影像，請檢查路徑。"

    shape = img_cv.shape
    has_color_content = False

    # 判斷物理通道
    if len(shape) == 2:
        physical_status = "單通道 (真正灰階檔)"
    elif len(shape) == 3:
        if shape[2] == 3:
            physical_status = "三通道 (RGB)"
            # 檢查像素內容：如果 R, G, B 三個通道值完全一樣，則是視覺灰階
            b, g, r = cv2.split(img_cv)
            if np.array_equal(b, g) and np.array_equal(g, r):
                has_color_content = False
            else:
                has_color_content = True
        elif shape[2] == 4:
            physical_status = "四通道 (RGBA)"
            # 同樣檢查 RGB 部分
            b, g, r, a = cv2.split(img_cv)
            has_color_content = not (np.array_equal(b, g) and np.array_equal(g, r))

    # --- 方法 B: 使用 Pillow 檢查模式 ---
    img_pil = Image.open(img_path)
    pil_mode = img_pil.mode # 'L' 為灰階, 'RGB' 為彩色

    print(f"檔案: {img_path}")
    print(f"物理形狀 (Shape): {shape}")
    print(f"Pillow 模式: {pil_mode}")
    print(f"結構狀態: {physical_status}")
    print(f"視覺內容: {'實質彩色 (有色偏)' if has_color_content else '實質灰階 (無色偏)'}")
    print("-" * 30)

if __name__ == "__main__":
# 測試
    path ='datasets/img/train/1.pgm'  # 替換為你的影像路徑
    check_image_channels(path)