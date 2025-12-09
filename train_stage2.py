import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.networks import EdgeGenerator, DiffusionUNet
from src.diffusion import DiffusionManager
from src.dataset import InpaintingDataset

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 2e-4
EPOCHS = 100
BATCH_SIZE = 8
G1_CHECKPOINT = "G1_epoch_19.pth" # 請確保此檔案存在

def train():
    dataset = InpaintingDataset("./datasets", mode='train')
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Models
    G1 = EdgeGenerator().to(DEVICE)
    # Load trained G1
    try:
        G1.load_state_dict(torch.load(G1_CHECKPOINT))
        print("G1 weights loaded.")
    except:
        print("Warning: G1 weights not found, using random init (NOT RECOMMENDED)")

    G1.eval() # Freeze G1
    for p in G1.parameters(): p.requires_grad = False

    G2 = DiffusionUNet().to(DEVICE)
    diffusion = DiffusionManager(device=DEVICE)

    opt = torch.optim.AdamW(G2.parameters(), lr=LR)
    mse = nn.MSELoss()

    print("--- Starting Stage 2 Training (Diffusion) ---")

    for epoch in range(EPOCHS):
        for i, (imgs, _, masks) in enumerate(loader):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            # 1. Generate Edges using frozen G1
            with torch.no_grad():
                # Input: Masked Image + Mask
                masked_imgs = imgs * (1 - masks)
                g1_input = torch.cat([masked_imgs, masks], dim=1)
                pred_edges = G1(g1_input)

            # 2. Prepare Diffusion Condition
            # Condition: Masked Image (3) + Mask (1) + Pred Edge (1) = 5 channels
            condition = torch.cat([masked_imgs, masks, pred_edges], dim=1)

            # 3. Diffusion Forward
            t = diffusion.sample_timesteps(imgs.shape[0])
            x_t, noise = diffusion.noise_images(imgs, t)

            # 4. Train G2 to predict noise
            opt.zero_grad()
            predicted_noise = G2(x_t, t, condition)

            loss = mse(noise, predicted_noise)
            loss.backward()
            opt.step()

            if i % 50 == 0:
                print(f"Epoch {epoch} [{i}/{len(loader)}] Diff Loss: {loss.item():.4f}")

        if epoch % 10 == 0:
            torch.save(G2.state_dict(), f"G2_diffusion_epoch_{epoch}.pth")

if __name__ == "__main__":
    train()