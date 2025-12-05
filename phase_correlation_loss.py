"""
phase_correlation_loss.py
Фазовая корреляция (POC) как лосс для стабилизации тайлинга.

Идея:
  - Убрать влияние амплитудных и краевых эффектов -> Ханново окно + вычитание среднего.
  - Сопоставлять только фазу: нормируем кросс-спектр по модулю (Phase-Only Correlation).
  - Максимизировать центральный пик корреляции (нулевой сдвиг) и PCE (одиночность пика).

Ссылки/обоснования:
  - POC (phase-only correlation) и классика метода: Kuglin & Hines (1975). 
  - Практика: OpenCV phaseCorrelate — Ханново окно, кросс-спектр, ifft (корр-карта).
  - PCE (peak-to-correlation energy) — стандартная метрика «одиночности» пика.

Пример использования
# начните с небольшого веса фазовой корреляции
loss = L_rec + L_spec + L_lp + L_acf + 0.05 * L_pcl
loss.backward()

# --- лог диагностики (раз в N шагов):
with torch.no_grad():
    diag = pcl.diagnostics(pred, target, mask)
    print(f"PCL diag: center_prob={diag['center_prob']:.3f}  PCE={diag['PCE']:.2f}"

"""
from __future__ import annotations
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

# берём утилиты из вашего пакета
from texture_helpers import _EPS, rgb_to_luma, feather


class PhaseCorrelationLoss(nn.Module):
    """
    Фазовая корреляция между pred и target внутри mask.

    Лосс =  center_weight * NLL(center)  +  pce_weight * (-log PCE)
      - NLL(center): отриц. лог-вероятность центральной ячейки корреляц. поверхности
                     (через softmax(corr/temperature))
      - PCE: peak^2 / mean_energy_outside, «одиночность» пика

    Рекомендуемый базовый вес в общей сумме лоссов: 0.02–0.1
    """
    def __init__(
        self,
        center_weight: float = 1.0,
        pce_weight: float = 0.1,
        temperature: float = 0.02,
        use_hann: bool = True,
        demean: bool = True,
        exclude_radius: int = 7,  # радиус центрального окна, исключаемого из энергии «шума»
    ):
        super().__init__()
        self.center_weight = float(center_weight)
        self.pce_weight = float(pce_weight)
        self.temperature = float(temperature)
        self.use_hann = bool(use_hann)
        self.demean = bool(demean)
        self.exclude_radius = int(exclude_radius)

    # ---------- публичная диагностика (для логов) ----------
    @torch.no_grad()
    def diagnostics(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
        """
        Возвращает:
          - center_prob: вероятность центральной ячейки softmax(corr/T)
          - PCE: peak-to-correlation-energy
        """
        Ia, Ib, m = self._prep_inputs(pred, target, mask)
        corr = self._phase_corr_map(Ia, Ib)  # Bx1xHxW
        center_prob = self._center_prob(corr)
        pce = self._pce_value(corr)
        return {"center_prob": float(center_prob.mean().cpu()), "PCE": float(pce.mean().cpu())}

    # ---------- интерфейс лосса ----------
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        Ia, Ib, m = self._prep_inputs(pred, target, mask)
        corr = self._phase_corr_map(Ia, Ib)  # Bx1xHxW

        L_center = self._nll_center(corr)
        L_pce    = self._pce_loss(corr)
        return self.center_weight * L_center + self.pce_weight * L_pce

    # ---------- внутренности ----------
    def _prep_inputs(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        Ia = rgb_to_luma(pred)   # Bx1xHxW
        Ib = rgb_to_luma(target) # Bx1xHxW
        m  = feather(mask, ksize=7).clamp(0, 1)
        Ia = self._apply_window(Ia, m)
        Ib = self._apply_window(Ib, m)
        return Ia, Ib, m

    def _hann2d(self, H, W, device, dtype):
        # 2D Ханна = outer(hann(H), hann(W)) — снижает краевые эффекты на DFT
        h = torch.hann_window(H, device=device, dtype=dtype, periodic=False)
        w = torch.hann_window(W, device=device, dtype=dtype, periodic=False)
        return (h[:, None] * w[None, :]).clamp_min(_EPS)  # HxW

    def _apply_window(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # аподизация маской, вычитание среднего в области маски, опц. окно Ханна
        xw = x * mask
        if self.demean:
            mean = (xw.sum(dim=(2, 3), keepdim=True) / (mask.sum(dim=(2, 3), keepdim=True) + _EPS))
            xw = (xw - mean) * mask
        if self.use_hann:
            B, _, H, W = x.shape
            hann = self._hann2d(H, W, x.device, x.dtype)[None, None, :, :]
            xw = xw * hann
        return xw

    def _phase_corr_map(self, Ia: torch.Tensor, Ib: torch.Tensor) -> torch.Tensor:
        """
        Ia, Ib: Bx1xHxW (после препроц.). Возвращает корреляционную карту Bx1xHxW
        (центр — нулевой сдвиг). Используем phase-only cross-power spectrum.
        """
        Fa = torch.fft.fft2(Ia.squeeze(1), norm="ortho")
        Fb = torch.fft.fft2(Ib.squeeze(1), norm="ortho")
        CPS = Fa * torch.conj(Fb)
        CPS = CPS / (torch.abs(CPS) + _EPS)      # Phase-Only Correlation (POC)
        corr = torch.fft.ifft2(CPS, norm="ortho").real
        corr = torch.fft.fftshift(corr, dim=(-2, -1))
        return corr.unsqueeze(1)

    # ----- центр -----
    def _center_prob(self, corr: torch.Tensor) -> torch.Tensor:
        B, _, H, W = corr.shape
        logits = corr.view(B, -1) / max(self.temperature, 1e-6)
        p = torch.softmax(logits, dim=-1)
        cy, cx = H // 2, W // 2
        idx = cy * W + cx
        return p[:, idx]  # B

    def _nll_center(self, corr: torch.Tensor) -> torch.Tensor:
        B, _, H, W = corr.shape
        logits = corr.view(B, -1) / max(self.temperature, 1e-6)
        logp = torch.log_softmax(logits, dim=-1)
        cy, cx = H // 2, W // 2
        idx = cy * W + cx
        return (-logp[:, idx]).mean()

    # ----- PCE -----
    def _pce_value(self, corr: torch.Tensor) -> torch.Tensor:
        B, _, H, W = corr.shape
        cy, cx = H // 2, W // 2
        yy, xx = torch.arange(H, device=corr.device), torch.arange(W, device=corr.device)
        YY, XX = torch.meshgrid(yy, xx, indexing="ij")
        dist2 = (YY - cy) ** 2 + (XX - cx) ** 2
        mask_out = (dist2 > (self.exclude_radius ** 2)).float()[None, None, :, :]  # 1x1xHxW
        peak = corr[:, :, cy, cx]                 # Bx1
        energy = ((corr ** 2) * mask_out).sum(dim=(2, 3)) / (mask_out.sum(dim=(2, 3)) + _EPS)
        pce = (peak ** 2) / (energy + _EPS)
        return pce.squeeze(1)  # B

    def _pce_loss(self, corr: torch.Tensor) -> torch.Tensor:
        pce = self._pce_value(corr)  # B
        return (-torch.log(pce + 1e-8)).mean()
