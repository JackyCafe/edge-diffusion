import torch

class DiffusionManager:
    def __init__(self, device="cuda", noise_steps=1000, beta_start=1e-4, beta_end=0.02):
        self.noise_steps = noise_steps
        self.device = device

        self.beta = torch.linspace(beta_start, beta_end, noise_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,), device=self.device)

    def sample(self, model, condition, n, steps=50):
        """
        支援 DDIM 快速採樣邏輯
        steps: 採樣疊代步數。
        """
        model.eval()
        with torch.no_grad():
            # 初始噪聲
            x = torch.randn((n, 3, condition.shape[2], condition.shape[3])).to(self.device)

            # 定義 DDIM 跳步序列
            times = torch.linspace(self.noise_steps - 1, 0, steps + 1).long().to(self.device)

            for i in range(steps):
                # 修正處：確保 torch.ones(n) 在建立時即指定 device
                t = (torch.ones(n, device=self.device) * times[i]).long()
                t_next = (torch.ones(n, device=self.device) * times[i+1]).long()

                # 預測噪聲
                predicted_noise = model(x, t, condition)

                alpha_hat = self.alpha_hat[t][:, None, None, None]
                alpha_hat_next = self.alpha_hat[t_next][:, None, None, None]

                # DDIM 公式：預測 x0
                pred_x0 = (x - torch.sqrt(1 - alpha_hat) * predicted_noise) / torch.sqrt(alpha_hat)

                # 計算下一個步驟的 x_t_next
                x = torch.sqrt(alpha_hat_next) * pred_x0 + torch.sqrt(1 - alpha_hat_next) * predicted_noise

        model.train()
        return x.clamp(-1, 1)