from guided_diffusion.guided_diffusion.unet import UNetModel
from torch import nn
import torch
from einops import repeat, rearrange
import math
class UnetLatent(nn.Module):
    def __init__(self):
        super(UnetLatent, self).__init__()
        self.channel = 4
        self.unet = UNetModel(
            image_size=(60, 80),
            in_channels=self.channel*2,
            model_channels=128,
            out_channels=self.channel,
            num_res_blocks=3,
            attention_resolutions=(4, 8),
            dropout=0,
            channel_mult=(1, 2, 4),
            conv_resample=True,
            dims=3,
            num_classes=None,
            task_tokens=True,
            task_token_channels=512,
            use_checkpoint=False,
            use_fp16=False,
            num_head_channels=32,
        )
        self.unet.convert_to_fp32()

    def forward(self, x, t, task_embed=None, **kwargs):
        '''
        Args:
            x: [b, 3f+3, h, w]
            t: [b,]
        '''
        f = x.shape[1] // self.channel - 1 
        x_cond = repeat(x[:, -self.channel:], 'b c h w -> b c f h w', f=f)
        x = rearrange(x[:, :-self.channel], 'b (f c) h w -> b c f h w', c=4)
        x = torch.cat([x, x_cond], dim=1)
        out = self.unet(x, t, task_embed, **kwargs)
        
        return rearrange(out, 'b c f h w -> b (f c) h w')
class UnetBridge(nn.Module):
    def __init__(self, dim=4):
        super(UnetBridge, self).__init__()
        self.unet = UNetModel(
            image_size=(48, 64),
            in_channels=6,
            model_channels=160,
            out_channels=3,
            num_res_blocks=3,
            attention_resolutions=(4, 8),
            dropout=0,
            channel_mult=(1, 2, 4),
            conv_resample=True,
            dims=3,
            num_classes=None,
            task_tokens=True,
            task_token_channels=512,
            use_checkpoint=False,
            use_fp16=False,
            num_head_channels=32,
        )
        self.unet.convert_to_fp32()

    def forward(self, x, t, task_embed=None, **kwargs):
        '''
        Args: 
            x: [b, 3f+3, h, w]
            t: [b,]
        '''
        f = x.shape[1] // 3 - 1 
        x_cond = repeat(x[:, -3:], 'b c h w -> b c f h w', f=f)
        x = rearrange(x[:, :-3], 'b (f c) h w -> b c f h w', c=3)
        x = torch.cat([x, x_cond], dim=1)
        out = self.unet(x, t, task_embed, **kwargs)
        return rearrange(out, 'b c f h w -> b (f c) h w')

class UnetMW(nn.Module):
    def __init__(self, depth=False):
        self.channel = 3 if not depth else 4
        self.depth = depth
        super(UnetMW, self).__init__()
        self.unet = UNetModel(
            image_size=(128, 128),
            in_channels=self.channel+3,
            model_channels=128,
            out_channels=self.channel,
            num_res_blocks=2,
            attention_resolutions=(8, 16),
            dropout=0,
            channel_mult=(1, 2, 3, 4, 5),
            conv_resample=True,
            dims=3,
            num_classes=None,
            task_tokens=True,
            task_token_channels=512,
            use_checkpoint=False,
            use_fp16=False,
            num_head_channels=32,
            use_scale_shift_norm=False,
        )
    def forward(self, x, t, task_embed=None, **kwargs):
        '''
        Args: 
            x: [b, (f c), h, w]
                (We just use RGB image as condition no matter depth or not)
            t: [b,]
        '''
        f = math.ceil(x.shape[1] / self.channel) - 1 
        x_cond = repeat(x[:, -3:], 'b c h w -> b c f h w', f=f)

        x = rearrange(x[:, :-3], 'b (f c) h w -> b c f h w', c=self.channel)
        x = torch.cat([x, x_cond], dim=1)
        out = self.unet(x, t, task_embed, **kwargs)
        return rearrange(out, 'b c f h w -> b (f c) h w')
      


