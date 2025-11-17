"""
texture_losses.py
Лоссы для стабилизации масштаба тайлинга при замене материала.

Зависимости: torch; внутренние утилиты из texture_helpers.py
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from texture_helpers import (
    _EPS,
    rgb_to_luma,
    fft_amp,
    radial_profile,
    soft_peak,
    log_polar_map,
    autocorr_map,
)

__all__ = [
    "SpectralPeriodLoss",
    "LogPolarAlignLoss",
    "ScaleConsistencyLoss",
    "ACFPeriodLoss",
]


class SpectralPeriodLoss(torch.nn.Module):
    """
    A) Попадание в основной частотный пик + узкополосное согласование спектра.
    Стабилизирует масштаб тайлинга, почти не трогая фактуру.
    """
    def __init__(self, r_bins: int = 256, tau: float = 0.06, band_sigma: float = 0.15,
                 w_peak: float = 1.0, w_band: float = 1.0,
                 min_bin: int = 2, max_bin: int | None = None):
        super().__init__()
        self.r_bins = r_bins
        self.tau = tau
        self.band_sigma = band_sigma
        self.w_peak = w_peak
        self.w_band = w_band
        self.min_bin = min_bin
        self.max_bin = max_bin

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                center_override: torch.Tensor | None = None) -> torch.Tensor:
        I = rgb_to_luma(pred)
        T = rgb_to_luma(target)
        A_pred = fft_amp(I, mask)
        A_tgt = fft_amp(T, mask)

        Sg = radial_profile(A_pred, r_bins=self.r_bins, min_bin=self.min_bin,
                            max_bin=self.max_bin if self.max_bin is not None else self.r_bins - 1)
        St = radial_profile(A_tgt, r_bins=self.r_bins, min_bin=self.min_bin,
                            max_bin=self.max_bin if self.max_bin is not None else self.r_bins - 1)

        fg_idx, fg = soft_peak(Sg, tau=self.tau, lo=self.min_bin,
                               hi=self.max_bin if self.max_bin is not None else self.r_bins - 1)
        ft_idx, ft = soft_peak(St, tau=self.tau, lo=self.min_bin,
                               hi=self.max_bin if self.max_bin is not None else self.r_bins - 1)

        l_peak = (torch.log(fg + _EPS) - torch.log(ft + _EPS)).abs().mean()

        if center_override is None:
            c = ft.detach()  # B, 0..1
        else:
            c = center_override.clamp(1e-3, 1 - 1e-3)

        grid = (torch.arange(self.r_bins, device=pred.device, dtype=pred.dtype) + 0.5) / self.r_bins
        w = torch.exp(-(torch.log(grid + _EPS)[None, :] - torch.log(c[:, None] + _EPS)) ** 2 / (2 * self.band_sigma ** 2))
        w = w / (w.sum(dim=-1, keepdim=True) + _EPS)

        l_band = (w * (Sg - St).abs()).sum(dim=-1).mean()

        return self.w_peak * l_peak + self.w_band * l_band


class LogPolarAlignLoss(torch.nn.Module):
    """
    B) Выравнивание лог-полярной карты амплитудных спектров (масштаб/угол -> сдвиги).
    """
    def __init__(self, out_r: int = 128, out_t: int = 180, r_min: float = 1e-2, w: float = 1.0):
        super().__init__()
        self.out_r, self.out_t, self.r_min, self.w = out_r, out_t, r_min, w

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        I = rgb_to_luma(pred)
        T = rgb_to_luma(target)
        A_pred = fft_amp(I, mask)
        A_tgt = fft_amp(T, mask)
        LPg = log_polar_map(A_pred, self.out_r, self.out_t, self.r_min)
        LPt = log_polar_map(A_tgt, self.out_r, self.out_t, self.r_min)
        L = F.mse_loss(torch.log(LPg + _EPS), torch.log(LPt + _EPS))
        return self.w * L


class ScaleConsistencyLoss(torch.nn.Module):
    """
    C) Консистентность масштаба между двумя стохастическими предсказаниями (одинаковые условия).
    """
    def __init__(self, r_bins: int = 256, tau: float = 0.06, min_bin: int = 2, max_bin: int | None = None, w: float = 1.0):
        super().__init__()
        self.r_bins = r_bins
        self.tau = tau
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.w = w

    def forward(self, pred1: torch.Tensor, pred2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        def _f(pred):
            I = rgb_to_luma(pred)
            A = fft_amp(I, mask)
            S = radial_profile(A, self.r_bins, self.min_bin,
                               self.max_bin if self.max_bin is not None else self.r_bins - 1)
            _, f = soft_peak(S, tau=self.tau, lo=self.min_bin,
                             hi=self.max_bin if self.max_bin is not None else self.r_bins - 1)
            return f

        f1 = _f(pred1)
        f2 = _f(pred2)
        return self.w * (torch.log(f1 + _EPS) - torch.log(f2 + _EPS)).abs().mean()


class ACFPeriodLoss(torch.nn.Module):
    """
    D) Период по автокорреляции: сравнивает положение первого пика ACF (радиальный профиль).
    """
    def __init__(self, r_bins: int = 256, tau: float = 0.08, min_rel: float = 0.03, max_rel: float = 0.6, w: float = 1.0):
        super().__init__()
        self.r_bins = r_bins
        self.tau = tau
        self.min_rel = min_rel
        self.max_rel = max_rel
        self.w = w

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        I = rgb_to_luma(pred); T = rgb_to_luma(target)
        Cg = autocorr_map(I, mask)  # Bx1xHxW
        Ct = autocorr_map(T, mask)
        Sg = radial_profile(Cg.abs(), r_bins=self.r_bins, min_bin=1, max_bin=self.r_bins - 1, log_scale=False)
        St = radial_profile(Ct.abs(), r_bins=self.r_bins, min_bin=1, max_bin=self.r_bins - 1, log_scale=False)
        lo = int(self.min_rel * (self.r_bins - 1))
        hi = int(self.max_rel * (self.r_bins - 1))
        _, fg = soft_peak(Sg, tau=self.tau, lo=lo, hi=hi)
        _, ft = soft_peak(St, tau=self.tau, lo=lo, hi=hi)
        return self.w * (torch.log(fg + _EPS) - torch.log(ft + _EPS)).abs().mean()
