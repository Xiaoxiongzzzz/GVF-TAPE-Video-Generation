import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
import wandb
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from diffusers.models import AutoencoderKL
from torch.optim import AdamW
from torchvision.utils import save_image
from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
import matplotlib.pyplot as plt
import numpy as np

__version__ = "0.0"

import os

from pynvml import *

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

import tensorboard as tb

# constants
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions
def tensors2vectors(tensors):
    def tensor2vector(tensor):
        flo = (tensor.permute(1, 2, 0).numpy()-0.5)*1000
        r = 8
        plt.quiver(flo[::-r, ::r, 0], -flo[::-r, ::r, 1], color='r', scale=r*20)
        plt.savefig('temp.jpg')
        plt.clf()
        return plt.imread('temp.jpg').transpose(2, 0, 1)
    return torch.from_numpy(np.array([tensor2vector(tensor) for tensor in tensors])) / 255

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


   
class GoalGaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        channels=3,
        timesteps = 1000,
        sampling_timesteps = 100,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = False,
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5
    ):
        super().__init__()

        self.model = model

        self.channels = channels

        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_cond, task_embed,  clip_x_start=False, rederive_pred_noise=False, guidance_weight=0):
        # task_embed = self.text_encoder(goal).last_hidden_state
        model_output = self.model(torch.cat([x, x_cond], dim=1), t, task_embed)
        if guidance_weight > 0.0:
            uncond_model_output = self.model(torch.cat([x, x_cond], dim=1), t, task_embed*0.0)

        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            if guidance_weight == 0:
                pred_noise = model_output
            else:
                pred_noise = (1 + guidance_weight)*model_output - guidance_weight*uncond_model_output

            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)

            if guidance_weight == 0:
                pred_noise = self.predict_noise_from_start(x, t, x_start)
            else:
                uncond_x_start = uncond_model_output
                uncond_x_start = maybe_clip(uncond_x_start)
                cond_noise = self.predict_noise_from_start(x, t, x_start)
                uncond_noise = self.predict_noise_from_start(x, t, uncond_x_start)
                pred_noise = (1 + guidance_weight)*cond_noise - guidance_weight*uncond_noise
                x_start = self.predict_start_from_noise(x, t, pred_noise)
            
        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            
            if guidance_weight == 0:
                pred_noise = self.predict_noise_from_start(x, t, x_start)
            else:
                uncond_v = uncond_model_output
                uncond_x_start = self.predict_start_from_v(x, t, uncond_v)
                uncond_noise = self.predict_noise_from_start(x, t, uncond_x_start)
                cond_noise = self.predict_noise_from_start(x, t, x_start)
                pred_noise = (1 + guidance_weight)*cond_noise - guidance_weight*uncond_noise
                x_start = self.predict_start_from_noise(x, t, pred_noise)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_cond, task_embed,  clip_denoised=False, guidance_weight=0):
        preds = self.model_predictions(x, t, x_cond, task_embed, guidance_weight=guidance_weight)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_cond, task_embed, guidance_weight=0):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x, batched_times, x_cond, task_embed, clip_denoised = True, guidance_weight=guidance_weight)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, x_cond, task_embed, return_all_timesteps=False, guidance_weight=0):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            # self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, x_cond, task_embed, guidance_weight=guidance_weight)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def ddim_sample(self, shape, x_cond, task_embed, return_all_timesteps=False, guidance_weight=0):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            # self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, x_cond, task_embed, clip_x_start = False, rederive_pred_noise = True, guidance_weight=guidance_weight)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def sample(self, x_cond, task_embed, batch_size = 16, return_all_timesteps = False, guidance_weight=0):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, image_size[0], image_size[1]), x_cond, task_embed,  return_all_timesteps = return_all_timesteps, guidance_weight=guidance_weight)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, x_cond, task_embed, noise=None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noisy sample

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # predict and take gradient step

        model_out = self.model(torch.cat([x, x_cond], dim=1), t, task_embed)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, img_cond, task_embed):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}, got({h}, {w})'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, img_cond, task_embed)

# trainer class
class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        tokenizer=None,  # 可以设为可选参数
        text_encoder=None,  # 可以设为可选参数
        train_set=None,
        valid_set=None,
        channels=3,
        *,
        train_batch_size=1,
        valid_batch_size=1,
        train_lr=1e-4,
        train_num_steps=100000,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=1000,
        num_samples=3,
        results_folder='./results',
        amp=True,
        fp16=True,
        split_batches=True,
        calculate_fid=True,
        inception_block_idx=2048,
        cond_drop_chance=0.0,
        use_wandb=False,
        depth=False,
        num_frames=6,
        sample_per_seq=7,
        interval=4,
        train_ratio=0.9,
        target_size=[128, 128],
        seed=None,
    ):
        super().__init__()
        # Save parameters
        self.use_wandb = use_wandb
        self.cond_drop_chance = cond_drop_chance
        self.depth = depth
        self.num_frames = num_frames
        self.sample_per_seq = sample_per_seq
        self.interval = interval
        self.train_ratio = train_ratio
        self.target_size = target_size
        self.seed = seed
        
        # Training related parameters
        self.lr = train_lr
        self.adam_betas = adam_betas
        self.train_num_steps = train_num_steps
        
        # Text encoding related - now optional
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
            
        # Setup accelerator
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no',
            log_with="wandb" if use_wandb else None
        )
        
        # Native amp setup
        self.accelerator.native_amp = amp

        # Model
        self.model = diffusion_model
        self.channels = channels

        # InceptionV3 for FID score calculation
        self.inception_v3 = None
        if calculate_fid:
            assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
            self.inception_v3 = InceptionV3([block_idx])
            self.inception_v3.to(self.device)

        # Sampling and training hyperparameters
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        self.batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.image_size = diffusion_model.image_size

        # Setup datasets and data loaders
        self.setup_datasets(train_set, valid_set)
        
        # Setup optimizer and scheduler
        self.setup_optimizer_and_scheduler()

        # Result saving directory
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)

        # Step counter
        self.step = 0

        # Prepare model and optimizer with accelerator
        to_prepare = [self.model, self.opt, self.scheduler]
        # 只有在提供了文本编码器的情况下才准备它
        if self.text_encoder is not None:
            to_prepare.append(self.text_encoder)
            prepared = self.accelerator.prepare(*to_prepare)
            self.model, self.opt, self.scheduler, self.text_encoder = prepared
            # Text encoder doesn't need gradients
            self.text_encoder.requires_grad_(False)
            self.text_encoder.eval()
        else:
            prepared = self.accelerator.prepare(*to_prepare)
            self.model, self.opt, self.scheduler = prepared
        
        # If using wandb, initialize accelerator's trackers
        if self.use_wandb:
            self.setup_wandb()

    def setup_wandb(self):
        """Setup wandb tracker"""
        if not self.accelerator.is_main_process:
            return
            
        full_config = {
            "model": {
                "depth": self.depth,
                "num_frames": self.num_frames,
                "learning_rate": self.lr,
                "train_steps": self.train_num_steps,
                "batch_size": self.batch_size,
                "num_gpus": self.accelerator.num_processes,
                "effective_batch": self.batch_size * self.accelerator.num_processes,
            },
            
            "data": {
                "target_size": self.target_size,
                "sample_per_seq": self.sample_per_seq,
                "interval": self.interval,
                "train_ratio": self.train_ratio,
            },
            
            "training": {
                "save_every": self.save_and_sample_every,
                "scheduler_T_max": self.train_num_steps * self.accelerator.num_processes,
                "seed": self.seed,
            }
        }
        
        self.accelerator.init_trackers(
            project_name="AVDC",
            config=full_config,
            init_kwargs={"wandb": {
                "name": f"goal-diffusion-{'rgbd' if self.depth else 'rgb'}-{self.accelerator.num_processes}gpus",
                "tags": ["multi-gpu", "goal-diffusion"],
            }}
        )

    def setup_datasets(self, train_set, valid_set):
        """Setup datasets and data loaders"""
        # Select samples for validation set
        valid_inds = [i for i in range(len(valid_set))][:self.num_samples]
        self.valid_set = Subset(valid_set, valid_inds)
        self.valid_set.augmentation = False

        # Setup training set
        self.train_set = train_set
        
        # Create data loaders
        train_dl = DataLoader(
            self.train_set, 
            batch_size=self.batch_size, 
            shuffle=True, 
            pin_memory=True, 
            num_workers=8
        )
        
        # Prepare training data loader
        self.train_dl = self.accelerator.prepare(train_dl)
        self.train_dl = cycle(self.train_dl)
        
        # Setup validation data loader
        self.valid_dl = DataLoader(
            self.valid_set, 
            batch_size=self.valid_batch_size, 
            shuffle=False, 
            pin_memory=True, 
            num_workers=8
        )
        
    def setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler"""
        self.opt = AdamW(self.model.parameters(), lr=self.lr, betas=self.adam_betas)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt,
            T_max=self.train_num_steps 
        )

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        """Save model checkpoint"""
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'optimizer': self.opt.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        # Ensure ckpt directory exists
        save_dir = self.results_folder / 'ckpt'
        save_dir.mkdir(exist_ok=True, parents=True)
        
        # Save model
        torch.save(data, str(save_dir / f'model_{milestone}.pt'))

    def load(self, milestone=None, checkpoint=None, fine_tune=False):
        """Load model checkpoint"""
        accelerator = self.accelerator
        device = accelerator.device
        
        # Load model based on provided milestone or checkpoint
        if milestone is not None:
            data = torch.load(str(self.results_folder / f'ckpt/model_{milestone}.pt'), map_location=device)
        else:
            data = torch.load(checkpoint, map_location=device)

        # Load model state
        model = self.accelerator.unwrap_model(self.model)
        model_checkpoint = data['model']
        
        # Handle module. prefix (do this for all processes, not just main)
        if any(k.startswith('module.') for k in model_checkpoint.keys()):
            model_checkpoint = {k.replace('module.', ''): v for k, v in model_checkpoint.items()}
        
        # 确保模型状态字典中的键匹配
        missing_keys, unexpected_keys = model.load_state_dict(model_checkpoint, strict=False)
        if missing_keys:
            print(f"Warning: Missing keys in state dict: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in state dict: {unexpected_keys}")
            
        # If not fine-tuning mode, restore training state
        if not fine_tune:
            self.step = data['step']
            print(f"Restored model state from step {self.step}")
            
            # Load optimizer state
            if "optimizer" in data:
                self.opt.load_state_dict(data["optimizer"])
                print("Restored optimizer state")
            
            # Load scheduler state
            if "scheduler" in data:
                self.scheduler.load_state_dict(data["scheduler"])
                print("Restored scheduler state")

        # Load scaler state
        if exists(self.accelerator.scaler) and exists(data.get('scaler')):
            self.accelerator.scaler.load_state_dict(data['scaler'])
            print("Restored scaler state")
            
        if 'version' in data:
            print(f"Loading from version {data['version']}")
            
        # 等待所有进程完成加载
        accelerator.wait_for_everyone()

    def sample(self, x_conds, task_embeds, batch_size=1, guidance_weight=0):
        """Generate samples using the model with provided task embeddings"""
        device = self.device
        
        # 获取未包装的原始模型
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        return unwrapped_model.sample(
            x_conds.to(device), 
            task_embeds.to(device), 
            batch_size=batch_size, 
            guidance_weight=guidance_weight
        )

    def train(self):
        """Train the model"""
        accelerator = self.accelerator
        device = accelerator.device

        # Create progress bar
        disable_progress_bar = not accelerator.is_local_main_process
        progress_bar = tqdm(
            range(self.step, self.train_num_steps),
            disable=disable_progress_bar
        )

        for step in progress_bar:
            # Get next batch of data
            x, x_cond, task_embed = next(self.train_dl)
            x, x_cond, task_embed = x.to(device), x_cond.to(device), task_embed.to(device)
            
            # Rearrange dimensions
            x = rearrange(x, "b f c h w -> b (f c) h w")
            
            # Apply condition dropping if needed
            if self.cond_drop_chance > 0:
                task_embed = task_embed * (torch.rand(task_embed.shape[0], 1, 1, device=task_embed.device) > self.cond_drop_chance).float()

            # Forward and backward pass
            with self.accelerator.autocast():
                loss = self.model(x, x_cond, task_embed)
                self.accelerator.backward(loss)

            # Gradient clipping
            accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

            # Get loss scale
            scale = self.accelerator.scaler.get_scale() if exists(self.accelerator.scaler) else 1.0
            
            # Update progress bar description
            progress_bar.set_postfix(loss=f"{loss.item():.4E}", loss_scale=f"{scale:.1E}")

            # Wait for all processes
            accelerator.wait_for_everyone()

            # Update model parameters
            self.opt.step()
            self.opt.zero_grad()

            # Update learning rate
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # Wait for all processes again
            accelerator.wait_for_everyone()

            # Increment step
            self.step += 1
            
            # Operations performed by main process
            if accelerator.is_main_process:
                # Log metrics
                if self.use_wandb:
                    self.accelerator.log({
                        'loss': loss.item(), 
                        'loss_scale': scale, 
                        'lr': current_lr
                    }, step=step)

                # Periodically sample and save
                if self.step % self.save_and_sample_every == 0:
                    self.sample_and_save(step)

            # Update progress bar
            progress_bar.update(1)

        self.sample_and_save(self.train_num_steps)
        accelerator.print('Training complete')

    @torch.no_grad()
    def sample_and_save(self, step, save_model=True):
        """Generate samples and save model"""
        # Only execute in main process
        if not self.accelerator.is_main_process:
            return
            
        # Set model to evaluation mode
        self.model.eval()
        
        # Load validation set samples
        xs = []
        x_conds = []
        task_embeds = []
        
        for x, x_cond, task_embed in self.valid_dl:
            xs.append(rearrange(x, "b f c h w -> b (f c) h w"))
            x_conds.append(x_cond)
            task_embeds.append(task_embed)  # 直接使用数据集提供的嵌入
        
        device = self.device
        batches = num_to_groups(self.num_samples, self.valid_batch_size)
        
        # Generate samples using the model
        with self.accelerator.autocast():
            # 获取未包装的原始模型
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            all_xs_list = list(map(
                lambda n, c, e: unwrapped_model.sample(
                    batch_size=n, 
                    x_cond=c.to(device), 
                    task_embed=e.to(device)  # 确保嵌入在正确的设备上
                ), 
                batches, x_conds, task_embeds
            ))
        
        # Process ground truth and generated samples
        gt_xs = torch.cat(xs, dim=0)
        n_rows = self.num_frames
        gt_xs = rearrange(gt_xs, 'b (n c) h w -> b n c h w', n=n_rows)

        x_conds = torch.cat(x_conds, dim=0).detach().cpu()
        all_xs = torch.cat(all_xs_list, dim=0)
        all_xs = rearrange(all_xs, 'b (n c) h w -> (b n) c h w', n=n_rows)
        all_xs = all_xs.detach().cpu()
        all_xs = rearrange(all_xs, '(b n) c h w -> b n c h w', n=n_rows)

        # Extract first and last frame for comparison
        gt_first = gt_xs[:, :1]
        gt_last = gt_xs[:, -1:]

        # Create save directory
        save_dir = os.path.join(self.results_folder, f"{'depth_' if self.depth else ''}imgs")
        os.makedirs(save_dir, exist_ok=True)
        
        # If RGBD mode, save RGB and depth images separately
        if self.depth:
            # Separate RGB and depth channels from generated samples
            rgb_sample = all_xs[:, :, :3]
            depth_sample = all_xs[:, :, 3:4]
            
            # Process RGB channels
            rgb_gt_start = gt_first[:, :, :3]  # First frame RGB
            rgb_gt_last = gt_last[:, :, :3]    # Last frame RGB
            
            # Get ground truth depth from last frame
            depth_gt_last = gt_last[:, :, 3:4]  # Extract depth channel from last frame
            
            # Concatenate RGB images (condition, target RGB, generated RGB)
            rgb_images = torch.cat([rgb_gt_start, rgb_gt_last, rgb_sample], dim=1)
            # Concatenate depth images (ground truth depth, generated depth)
            depth_images = torch.cat([depth_gt_last, depth_sample], dim=1)
            
            # Prepare for visualization
            n_row_rgb = rgb_images.shape[1]
            n_row_depth = depth_images.shape[1]
            rgb_images = rearrange(rgb_images, "b f c h w -> (b f) c h w")
            depth_images = rearrange(depth_images, "b f c h w -> (b f) c h w")
            
            # Save images
            save_image(rgb_images, os.path.join(save_dir, f"sample_rgb_{step}.png"), nrow=n_row_rgb)
            save_image(depth_images, os.path.join(save_dir, f"sample_depth_{step}.png"), nrow=n_row_depth)
        else:
            # Process RGB images only
            images = torch.cat([gt_first, gt_last, all_xs], dim=1)
            n_row = images.shape[1]
            images = rearrange(images, "b f c h w -> (b f) c h w")
            save_image(images, os.path.join(save_dir, f"sample_{step}.png"), nrow=n_row)

        # Return model to training mode
        self.model.train()
        
        # Save model checkpoint
        if save_model:
            self.save(step)