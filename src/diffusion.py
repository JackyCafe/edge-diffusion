import torch

class DiffusionManager:
    def __init__(self, device="cuda", noise_steps=1000, beta_start=1e-4, beta_end=0.02):
        self.noise_steps = noise_steps
        self.device = device

        # 定義線性噪聲排程
        self.beta = torch.linspace(beta_start, beta_end, noise_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def noise_images(self, x, t):
        """ 訓練時的前向加噪過程 """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        """ 隨機採樣時間步 """
        return torch.randint(low=1, high=self.noise_steps, size=(n,), device=self.device)

    def sample(self, model, condition, n, steps=50):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, condition.shape[2], condition.shape[3])).to(self.device)
            times = torch.linspace(self.noise_steps - 1, 0, steps + 1).long().to(self.device)

            for i in range(steps):
                t = (torch.ones(n, device=self.device) * times[i]).long()
                t_next = (torch.ones(n, device=self.device) * times[i+1]).long()

                predicted_noise = model(x, t, condition)
                # [防線 1] 限制噪聲預測範圍
                predicted_noise = torch.clamp(predicted_noise, -1.5, 1.5)

                alpha_hat = self.alpha_hat[t][:, None, None, None]
                alpha_hat_next = self.alpha_hat[t_next][:, None, None, None]

                # [防線 2] 鎖定分母最小值，防止數值爆炸
                safe_sqrt_alpha = torch.sqrt(alpha_hat).clamp(min=0.2)

                pred_x0 = (x - torch.sqrt(1 - alpha_hat) * predicted_noise) / safe_sqrt_alpha
                pred_x0 = torch.clamp(pred_x0, -1, 1)

                x = torch.sqrt(alpha_hat_next) * pred_x0 + torch.sqrt(1 - alpha_hat_next) * predicted_noise
        model.train()
        return x.clamp(-1.5, 1.5)
    
