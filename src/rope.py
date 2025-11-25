import math

import numpy as np
import torch


def find_correction_factor(num_rotations, dim, base, max_position_embeddings):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def find_correction_range(low_ratio, high_ratio, dim, base, ori_max_pe_len):
    low = np.floor(find_correction_factor(low_ratio, dim, base, ori_max_pe_len))
    high = np.ceil(find_correction_factor(high_ratio, dim, base, ori_max_pe_len))
    return max(low, 0), min(high, dim - 1)


def linear_ramp_mask(min_val, max_val, dim):
    if min_val == max_val:
        max_val += 0.001

    linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def find_newbase_ntk(dim, base, scale):
    return base * (scale ** (dim / (dim - 2)))


@torch.no_grad()
def get_1d_rotary_pos_embed(
    dim: int,
    pos: torch.Tensor,
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype=torch.float32,
    yarn=False,
    max_pe_len=None,
    ori_max_pe_len=64,
    dype=False,
    current_timestep=1.0,
    dype_exponent=2.0,
):
    assert dim % 2 == 0
    device = pos.device

    if yarn and max_pe_len is not None and max_pe_len > ori_max_pe_len:
        if not isinstance(max_pe_len, torch.Tensor):
            max_pe_len = torch.tensor(max_pe_len, dtype=freqs_dtype, device=device)

        scale = torch.clamp_min(max_pe_len / ori_max_pe_len, 1.0)

        beta_0, beta_1 = 1.25, 0.75
        gamma_0, gamma_1 = 16, 2

        freqs_base = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=device) / dim))
        freqs_linear = 1.0 / torch.einsum('..., f -> ... f', scale, (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=device) / dim)))

        new_base = find_newbase_ntk(dim, theta, scale)
        if new_base.dim() > 0:
            new_base = new_base.view(-1, 1)
        freqs_ntk = 1.0 / torch.pow(new_base, (torch.arange(0, dim, 2, dtype=freqs_dtype, device=device) / dim))
        if freqs_ntk.dim() > 1:
            freqs_ntk = freqs_ntk.squeeze()

        if dype:
            beta_0 = beta_0 ** (dype_exponent * (current_timestep ** dype_exponent))
            beta_1 = beta_1 ** (dype_exponent * (current_timestep ** dype_exponent))

        low, high = find_correction_range(beta_0, beta_1, dim, theta, ori_max_pe_len)
        low, high = max(0, low), min(dim // 2, high)

        freqs_mask = (1 - linear_ramp_mask(low, high, dim // 2).to(device).to(freqs_dtype))
        freqs = freqs_linear * (1 - freqs_mask) + freqs_ntk * freqs_mask

        if dype:
            gamma_0 = gamma_0 ** (dype_exponent * (current_timestep ** dype_exponent))
            gamma_1 = gamma_1 ** (dype_exponent * (current_timestep ** dype_exponent))

        low, high = find_correction_range(gamma_0, gamma_1, dim, theta, ori_max_pe_len)
        low, high = max(0, low), min(dim // 2, high)

        freqs_mask = (1 - linear_ramp_mask(low, high, dim // 2).to(device).to(freqs_dtype))
        freqs = freqs * (1 - freqs_mask) + freqs_base * freqs_mask

    else:
        theta_ntk = theta * ntk_factor
        if dype and ntk_factor > 1.0:
            theta_ntk = theta * (ntk_factor ** (dype_exponent * (current_timestep ** dype_exponent)))

        freqs = 1.0 / (theta_ntk ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=device) / dim)) / linear_factor

    freqs = torch.einsum("...s,d->...sd", pos, freqs)

    if use_real and repeat_interleave_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=-1).float()
        freqs_sin = freqs.sin().repeat_interleave(2, dim=-1).float()

        if yarn and max_pe_len is not None and max_pe_len > ori_max_pe_len:
            mscale = torch.where(scale <= 1.0, torch.tensor(1.0), 0.1 * torch.log(scale) + 1.0).to(scale)
            freqs_cos, freqs_sin = freqs_cos * mscale, freqs_sin * mscale
        return freqs_cos, freqs_sin
    elif use_real:
        return freqs.cos().float(), freqs.sin().float()
    else:
        return torch.polar(torch.ones_like(freqs), freqs)
