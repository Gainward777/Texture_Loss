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
"""
Комментрий на основе эмпирического опыта
В качестве основы используем SpectralPeriodLoss. Именно данный тип лоссов наиболее сильно влияет на обучение. 
Помимо этого с малыми весами также используем LogPolarAlignLoss и ACFPeriodLoss. Если их дополнительно добавляем к SpectralPeriodLoss,
помогает модели при обучении сохранять тайтлинг. Очень было заметно это.

При обучении текстурные лоссы добавляю к основным с переменным коэффициентом. Пока это на стадии тестирования. Не могу ничего точно сказать
как и что лучше себя показывает.
"""

class SpectralPeriodLoss(torch.nn.Module):
    """
    A) Попадание в основной частотный пик + узкополосное согласование спектра.
    Стабилизирует масштаб тайлинга, почти не трогая фактуру.
    """

    # Комментарий от Дмитрия:
    # Очень классно проработанная функция подсчёта лоссов, которая хорошо себя показывает.
    # Её используем в качестве основы для текстурных лоссов, удобно играться с параметрами.

    def __init__(
        self,
        r_bins: int = 256,
        tau: float = 0.06,
        band_sigma: float = 0.15,
        w_peak: float = 1.0,
        w_band: float = 1.0,
        min_bin: int = 2,
        max_bin: int | None = None,
    ):
        super().__init__()
        self.r_bins = r_bins
        self.tau = tau
        self.band_sigma = band_sigma
        self.w_peak = w_peak
        self.w_band = w_band
        self.min_bin = min_bin
        self.max_bin = max_bin

        grid = (torch.arange(r_bins).float() + 0.5) / r_bins  # (R,)
        self.register_buffer("grid", grid)

    def _max_bin(self) -> int:
        return self.max_bin if self.max_bin is not None else self.r_bins - 1

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        center_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        #experimental correcting
        pred_0 = (pred / 2 + 0.5).clamp(0, 1)
        tgt_0  = (target / 2 + 0.5).clamp(0, 1)
        # Переводим в яркость
        I = rgb_to_luma(pred_0)
        T = rgb_to_luma(tgt_0)

        # Амплитудные спектры
        A_pred = fft_amp(I, mask)
        A_tgt = fft_amp(T, mask)

        max_bin = self._max_bin()

        # Радиальные профили спектров
        Sg = radial_profile(
            A_pred,
            r_bins=self.r_bins,
            min_bin=self.min_bin,
            max_bin=max_bin,
        )
        St = radial_profile(
            A_tgt,
            r_bins=self.r_bins,
            min_bin=self.min_bin,
            max_bin=max_bin,
        )

        # Мягкий поиск основного частотного пика
        _, fg = soft_peak(Sg, tau=self.tau, lo=self.min_bin, hi=max_bin)
        _, ft = soft_peak(St, tau=self.tau, lo=self.min_bin, hi=max_bin)

        # Лосс по пику (по лог-амплитуде)
        l_peak = (torch.log(fg + _EPS) - torch.log(ft + _EPS)).abs().mean()

        # Центр полосы: либо таргетный пик, либо явно заданный override
        if center_override is None:
            c = ft.detach()  # (B,), в [0, 1]
        else:
            c = center_override
        c = c.clamp(1e-3, 1 - 1e-3)

        # Гауссовское окно в лог-пространстве частот вокруг центра c
        grid = self.grid.to(dtype=pred.dtype)  # (R,)
        log_grid = torch.log(grid + _EPS)[None, :]        # (1, R)
        log_c = torch.log(c + _EPS)[:, None]              # (B, 1)

        w = torch.exp(-(log_grid - log_c) ** 2 / (2 * self.band_sigma ** 2))  # (B, R)
        w = w / (w.sum(dim=-1, keepdim=True) + _EPS)

        # Узкополосное согласование спектров
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
        #experimental correcting
        pred_0 = (pred / 2 + 0.5).clamp(0, 1)
        tgt_0  = (target / 2 + 0.5).clamp(0, 1)
        I = rgb_to_luma(pred_0)
        T = rgb_to_luma(tgt_0)
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
            #experimental correcting
            pred_0 = (pred / 2 + 0.5).clamp(0, 1)
            
            I = rgb_to_luma(pred_0)
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
        #experimental correcting
        pred_0 = (pred / 2 + 0.5).clamp(0, 1)
        tgt_0  = (target / 2 + 0.5).clamp(0, 1)
        I = rgb_to_luma(pred_0); T = rgb_to_luma(tgt_0)
        Cg = autocorr_map(I, mask)  # Bx1xHxW
        Ct = autocorr_map(T, mask)
        Sg = radial_profile(Cg.abs(), r_bins=self.r_bins, min_bin=1, max_bin=self.r_bins - 1, log_scale=False)
        St = radial_profile(Ct.abs(), r_bins=self.r_bins, min_bin=1, max_bin=self.r_bins - 1, log_scale=False)
        lo = int(self.min_rel * (self.r_bins - 1))
        hi = int(self.max_rel * (self.r_bins - 1))
        _, fg = soft_peak(Sg, tau=self.tau, lo=lo, hi=hi)
        _, ft = soft_peak(St, tau=self.tau, lo=lo, hi=hi)
        return self.w * (torch.log(fg + _EPS) - torch.log(ft + _EPS)).abs().mean()
