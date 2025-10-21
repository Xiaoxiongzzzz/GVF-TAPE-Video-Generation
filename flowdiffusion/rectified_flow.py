import torch
from einops import rearrange
class RectifiedFlow:
    def __init__(
                self,
                sample_timestep,
                sample_method="uniform"
                ):
        
        assert sample_method in ["uniform"]
        self.sample_method = sample_method
        self.sample_timestep = sample_timestep

    def sample(self, model, noise, x_cond, y):
        """
        Sample from the model.
        Args:
            model: nn.module
            noise: shape = [b, (f, c), h, w]
            x_cond: shape = [b, c, h, w]
            y: shape = [b, c] language embeding
        Return:
            x_0: shape = [b, (f, c), h, w]
        """
        B = noise.shape[0]
        times = [t/self.sample_timestep for t in range(self.sample_timestep)]
        delta_t = 1/self.sample_timestep
        
        x_noisy = noise
        for t in times:
            x_input = torch.cat([x_noisy, x_cond], dim=1)
            noise_level = torch.Tensor([1 - t] * B).to(noise.device)
            vel_pred = model(x_input, noise_level, y)
            x_noisy = x_noisy + vel_pred * delta_t
        x_0 = x_noisy
        return x_0

    def train_loss(self, model, x_start, x_cond, y, t=None):
        """
        Compute the training losses for the model.
        Args:
            model: nn.module
            x_start: shape = [b, (f, c), h, w]    
            x_cond: shape = [b, c(3), h, w] 
                    (We just use RGB image as condition no matter depth or not)
            y: shape = [b, c]   language embedding 
            t: float(0 - 1) or None
        """
        B = x_start.shape[0]
        if t is None:
            if self.sample_method == "uniform":
                t = torch.rand((B,), device=x_start.device, dtype=x_start.dtype)
        else:
            t = torch.full((B,), t, device=x_start.device, dtype=x_start.dtype)

        x_noisy, noise = self.add_noise(x_start, t)
        x_input = torch.cat([x_noisy, x_cond], dim=1)
        
        noise_level = (1 - t).to(dtype=x_start.dtype)  # 确保数据类型匹配
        vel_pred = model(x_input, noise_level, y)
        vel = x_start - noise
        
        loss = torch.mean((vel_pred - vel).pow(2))

        return loss

    def add_noise(
                self,
                x_start: torch.Tensor,
                t: torch.Tensor
                ):
        """
        Add noise to the input image.
        Args:
            x_start: shape = [b, (f, c), h, w]
            t: shape = [b,] t=0 means prior distribution t=1 means data distribution
        """
        t = t.view(-1, 1, 1, 1)  # 保持维度变换
        noise = torch.randn_like(x_start, device=x_start.device, dtype=x_start.dtype)
        
        noisy_x = t * x_start + (1 - t) * noise
        
        return noisy_x, noise
# # Test code
# if __name__ == "__main__":
#     from unet import UnetLatent as Unet
#     device = torch.device("cuda")
#     unet = Unet()
#     rectified_flow = RectifiedFlow(sample_timestep=10)
#     x_start = torch.randn(1, 24, 16, 16)
#     x_cond = torch.randn(1, 4, 16, 16)
#     y = torch.randn(1, 21, 512)
#     loss = rectified_flow.sample(unet, x_start, x_cond, y)
#     import ipdb;ipdb.set_trace()
