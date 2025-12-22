"""
texture_helpers.py
Вспомогательные функции: загрузка PNG в тензоры, маски/глубина,
и спектральные утилиты для лоссов.

Зависимости: torch, numpy, Pillow (PIL)
"""
from __future__ import annotations

import os
from math import pi
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

__all__ = [
    "load_png_rgb",
    "load_png_gray",
    "load_png_depth",
    "load_optional_mask",
    "rgb_to_luma",
    "feather",
    "fft_amp",
    "radial_profile",
    "soft_peak",
    "autocorr_map",
    "log_polar_map",
]

_EPS = 1e-8


# ------------------------- I/O: PNG -> tensors ------------------------- #

def _to_float01(t: torch.Tensor) -> torch.Tensor:
    """Привести uint8/uint16/int32/float к float32 [0..1]."""
    if t.dtype == torch.uint8:
        return t.float() / 255.0
    if t.dtype == torch.int32:
        # Pillow иногда читает 16-бит PNG как int32 ("I" mode).
        if int(t.max()) <= 65535 and int(t.min()) >= 0:
            t = t.to(torch.uint16)
        else:
            return ((t - t.min()).float() / (t.max() - t.min() + _EPS)).clamp(0, 1)
    if t.dtype == torch.uint16:
        return t.float() / 65535.0
    return t.float().clamp(0, 1)


def _hwc_numpy_from_pil(img: Image.Image) -> np.ndarray:
    """PIL.Image -> ndarray (H, W, C); gray -> (H, W, 1)."""
    arr = np.array(img)  # dtype сохранится (u8/u16/i32/float)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    return arr


def _chw01_from_pil(img: Image.Image, force_mode: Optional[str] = None) -> torch.Tensor:
    """
    Вернуть CxHxW float32 в [0..1]. force_mode: 'RGB'|'L'|None.
    """
    if force_mode is not None:
        img = img.convert(force_mode)
    arr = _hwc_numpy_from_pil(img)             # HxWxC
    t = torch.from_numpy(arr).permute(2, 0, 1) # CxHxW
    t = _to_float01(t)
    return t


def _resize_like(x: torch.Tensor, size_hw: Tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
    """Изменение размера CxHxW -> CxH'xW'."""
    x4 = x.unsqueeze(0) if x.dim() == 3 else x
    y4 = F.interpolate(
        x4, size=size_hw, mode=mode,
        align_corners=False if mode in ("bilinear", "bicubic", "trilinear", "linear") else None
    )
    return y4.squeeze(0) if x.dim() == 3 else y4


def load_png_rgb(path: str, size_hw: Optional[Tuple[int, int]] = None, device: str = "cpu") -> torch.Tensor:
    """
    Загрузка цветного PNG как 1x3xHxW float32 [0..1].
    """
    img = Image.open(path).convert("RGB")
    t = _chw01_from_pil(img)  # 3xHxW
    if size_hw is not None and (t.shape[1], t.shape[2]) != tuple(size_hw):
        t = _resize_like(t, size_hw, mode="bilinear")
    return t.unsqueeze(0).to(device)


def load_png_gray(path: str, size_hw: Optional[Tuple[int, int]] = None, device: str = "cpu",
                  nearest: bool = True) -> torch.Tensor:
    """
    Загрузка 1-канального PNG (маски/кромки) как 1x1xHxW float32 [0..1].
    """
    img = Image.open(path)
    if img.mode in ("LA", "RGBA"):
        img = img.split()[0]  # берём первый канал
    else:
        img = img.convert("L")
    t = _chw01_from_pil(img)  # 1xHxW
    if size_hw is not None and (t.shape[1], t.shape[2]) != tuple(size_hw):
        t = _resize_like(t, size_hw, mode="nearest" if nearest else "bilinear")
    return t.unsqueeze(0).to(device)


def load_png_depth(path: str, size_hw: Optional[Tuple[int, int]] = None, device: str = "cpu",
                   clamp01: bool = True) -> torch.Tensor:
    """
    Загрузка карты глубины PNG как 1x1xHxW float32.
    Поддержка 8/16-бит (в т.ч. Pillow mode 'I').
    """
    img = Image.open(path)  # 'I;16'/'I'/'L'/'F' и т.д.
    arr = _hwc_numpy_from_pil(img)  # HxWx1
    t = torch.from_numpy(arr).permute(2, 0, 1)  # 1xHxW
    t = _to_float01(t) if clamp01 else t.float()
    if size_hw is not None and (t.shape[1], t.shape[2]) != tuple(size_hw):
        t = _resize_like(t, size_hw, mode="bilinear")
    return t.unsqueeze(0).to(device)


def load_optional_mask(mask_path: Optional[str], fallback_size_hw: Tuple[int, int], device: str = "cpu") -> torch.Tensor:
    """
    Загрузка бинарной маски 1x1xHxW; если пути нет — ones.
    """
    if mask_path and os.path.exists(mask_path):
        m = load_png_gray(mask_path, size_hw=fallback_size_hw, device=device, nearest=True)  # 1x1xHxW
        return (m >= 0.5).float()
    H, W = fallback_size_hw
    return torch.ones(1, 1, H, W, device=device, dtype=torch.float32)


# ------------------------- Spectral utilities -------------------------- #

def rgb_to_luma(x: torch.Tensor) -> torch.Tensor:
    """Bx3xHxW -> Bx1xHxW (BT.2020-like)."""
    if x.size(1) == 1:
        return x
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.2627 * r + 0.6780 * g + 0.0593 * b


def feather(mask: torch.Tensor, ksize: int = 7) -> torch.Tensor:
    """Оперение (аподизация) маски для снижения утечек спектра."""
    if ksize <= 1:
        return mask
    pad = ksize // 2
    ker1d = torch.arange(ksize, device=mask.device, dtype=mask.dtype) - pad
    ker1d = torch.exp(-(ker1d ** 2) / (2 * (ksize / 3.0) ** 2))
    ker1d = (ker1d / ker1d.sum()).view(1, 1, 1, -1)
    m = F.conv2d(mask, ker1d, padding=(0, pad))
    m = F.conv2d(m, ker1d.transpose(2, 3), padding=(pad, 0))
    return m.clamp(0, 1)


def fft_amp(img: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            normalize: bool = True) -> torch.Tensor:
    """
    Амплитуда 2D-FFT (fftshift). img: Bx1xHxW -> Bx1xHxW (амплитуда).
    """
    if normalize:
        img = normalize_for_fft(img, mask)
    else:
        if mask is not None:
            img = img * feather(mask, ksize=7)

    F2 = torch.fft.fft2(img.squeeze(1), norm="ortho")
    F2 = torch.fft.fftshift(F2, dim=(-2, -1))
    A = (F2.real ** 2 + F2.imag ** 2 + _EPS).sqrt()
    return A.unsqueeze(1)



def radial_profile(mag: torch.Tensor, r_bins: int = 256, min_bin: int = 2,
                   max_bin: Optional[int] = None, log_scale: bool = True) -> torch.Tensor:
    """
    Радиально усреднённый спектр. mag: Bx1xHxW -> BxR
    """
    B, _, H, W = mag.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=mag.device),
        torch.linspace(-1, 1, W, device=mag.device),
        indexing="ij",
    )
    rr = torch.sqrt(xx ** 2 + yy ** 2)
    rr = rr / rr.max()  # 0..1
    if max_bin is None:
        max_bin = r_bins - 1
    r = rr * (r_bins - 1)
    r_idx = r.long().clamp(0, r_bins - 1)

    S = torch.zeros(B, r_bins, device=mag.device, dtype=mag.dtype)
    C = torch.zeros(B, r_bins, device=mag.device, dtype=mag.dtype)
    flat_idx = r_idx.view(-1)
    one = torch.ones_like(flat_idx, dtype=mag.dtype)
    for b in range(B):
        v = mag[b, 0].reshape(-1)
        S[b].scatter_add_(0, flat_idx, v)
        C[b].scatter_add_(0, flat_idx, one)
    S = S / (C + _EPS)
    if log_scale:
        S = torch.log(S + _EPS)
    # пригладим DC и хвост
    S[:, :min_bin] = S[:, min_bin:min_bin + 1]
    S[:, max_bin + 1:] = S[:, max_bin:max_bin + 1]
    return S


def soft_peak(freq_curve: torch.Tensor, tau: float = 0.1, lo: int = 0, hi: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Дифференцируемый soft-argmax пика по диапазону [lo..hi].
    Возвращает: (индекс-пик B,), (нормированная частота 0..1 B,)
    """
    B, R = freq_curve.shape
    if hi is None:
        hi = R - 1
    window = torch.zeros_like(freq_curve)
    window[:, lo:hi + 1] = 1.0
    z = freq_curve - freq_curve.max(dim=-1, keepdim=True).values
    p = torch.softmax(z / tau, dim=-1) * window
    p = p / (p.sum(dim=-1, keepdim=True) + _EPS)
    idxs = torch.arange(R, device=freq_curve.device, dtype=freq_curve.dtype)
    f_hat = (p * idxs).sum(dim=-1)  # B
    f_hat_norm = (f_hat + 0.5) / R
    return f_hat, f_hat_norm


def autocorr_map(img: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    ACF через Винера–Хинчина: ifft(PSD). img: Bx1xHxW -> Bx1xHxW (ACF)
    """
    A = fft_amp(img, mask)             # амплитуда
    P = (A ** 2).clamp_min(_EPS)       # спектр мощности
    P = torch.fft.ifftshift(P.squeeze(1), dim=(-2, -1))
    acf = torch.fft.ifft2(P, norm="ortho").real
    acf = torch.fft.fftshift(acf, dim=(-2, -1))
    return acf.unsqueeze(1)


def log_polar_map(mag: torch.Tensor, out_r: int = 128, out_t: int = 180, r_min: float = 1e-2) -> torch.Tensor:
    """
    Лог-полярное отображение амплитуды спектра.
    mag: Bx1xHxW (после fftshift) -> Bx1x(out_r)x(out_t)
    """
    B, _, H, W = mag.shape
    u = torch.linspace(0, 1, out_r, device=mag.device, dtype=mag.dtype)
    theta = torch.linspace(-pi, pi, out_t, device=mag.device, dtype=mag.dtype)
    r = r_min * (1.0 / r_min) ** u  # лог-радиусы в [r_min..1]

    rr, tt = torch.meshgrid(r, theta, indexing="ij")  # out_r x out_t
    x = rr * torch.cos(tt)
    y = rr * torch.sin(tt)
    grid = torch.stack([x, y], dim=-1)  # out_r x out_t x 2 (x, y) в [-1,1]
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)

    # grid_sample: координаты нормализованы в [-1,1]
    lp = F.grid_sample(
        mag, grid, mode="bilinear", padding_mode="zeros",
        align_corners=True  # фиксируем для стабильной геометрии
    )
    return lp



def normalize_for_fft(img: torch.Tensor,
                      mask: Optional[torch.Tensor] = None,
                      eps: float = _EPS,
                      do_mean: bool = True,
                      do_rms: bool = True) -> torch.Tensor:
    """
    Нормализация перед FFT:
    - вычитание среднего (DC removal)
    - деление на RMS (контраст), чтобы спектр меньше зависел от освещения/экспозиции
    Всё дифференцируемо.
    img: Bx1xHxW
    mask: Bx1xHxW (опционально)
    """
    if mask is None:
        x = img
        if do_mean:
            x = x - x.mean(dim=(-2, -1), keepdim=True)
        if do_rms:
            rms = x.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt()
            x = x / (rms + eps)
        return x

    # Важно: используем аподизацию как "веса" для статистик
    w = feather(mask, ksize=7)
    denom = w.sum(dim=(-2, -1), keepdim=True).clamp_min(eps)

    if do_mean:
        mu = (img * w).sum(dim=(-2, -1), keepdim=True) / denom
        x = img - mu
    else:
        x = img

    if do_rms:
        var = (x.pow(2) * w).sum(dim=(-2, -1), keepdim=True) / denom
        x = x / (var.sqrt() + eps)

    # И только после нормализации применяем окно (как вы и делали)
    return x * w

