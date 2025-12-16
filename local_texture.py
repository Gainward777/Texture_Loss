# local_texture_regularizer.py
from __future__ import annotations
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from toolkit.util.texture_losses import SpectralPeriodLoss, LogPolarAlignLoss, ScaleConsistencyLoss, ACFPeriodLoss
from toolkit.util.phase_correlation_loss import PhaseCorrelationLoss
from toolkit.util.texture_helpers import _EPS


def extract_patches(
    x: torch.Tensor,
    patch_size: int,
    stride: int,
) -> torch.Tensor:
    """
    x: BxCxHxW
    return: (B * N_patches) x C x patch_size x patch_size
    """
    B, C, H, W = x.shape
    p = patch_size

    patches = (
        x.unfold(2, p, stride)   
         .unfold(3, p, stride)   
    )
    B, C, Nh, Nw, _, _ = patches.shape
    patches = patches.contiguous().view(B * Nh * Nw, C, p, p)
    return patches


def extract_masked_patches(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    patch_size: int,
    stride: int,
    min_coverage: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Разрезает pred/target/mask на патчи и оставляет только те,
    где доля маски >= min_coverage.

    pred, target: BxCxHxW
    mask: Bx1xHxW (0..1)
    """
    B, _, H, W = pred.shape
    p = patch_size

    pred_p = extract_patches(pred, p, stride)
    tgt_p  = extract_patches(target, p, stride)
    m_p    = extract_patches(mask, p, stride)  # B*N x 1 x p x p

    # Покрытие маски в каждом патче
    area = p * p
    cover = m_p.view(m_p.size(0), -1).mean(dim=1)  # B*N

    keep = cover >= min_coverage
    if keep.any():
        pred_p = pred_p[keep]
        tgt_p  = tgt_p[keep]
        m_p    = m_p[keep]
    else:
        pred_p = pred_p[:1]
        tgt_p  = tgt_p[:1]
        m_p    = m_p[:1]

    return pred_p, tgt_p, m_p


class LocalTextureRegularityLoss(nn.Module):
    """
    Локальный лосс регулярности текстуры для сцен с перспективой.

    Идея:
      - режем область (по маске) на патчи patch_size x patch_size с шагом stride
      - на каждом патче считаем:
          * SpectralPeriodLoss 
          * ACFPeriodLoss      
          * PhaseCorrelationLoss 
      - усредняем по всем патчам

    Это не фиксирует глобальный период, но стабилизирует локальную
    структуру кирпичей при разном масштабе и перспективе.

    Использование:
        loss_tex = local_tex(pred, target, mask)
        L_total = mse + lambda_tex * loss_tex
    """

    def __init__(
        self,
        patch_size: int = 64,
        stride: int = 32,
        min_coverage: float = 0.3,
        # веса внутри комбинированного лосса
        w_spectral: float = 1.0,
        w_afc: float = 1.0,
        w_phase: float = 0.5,
        w_log: float = 0.2,

        #----------Phase--------------
        phase_center_weight: float = 1.0,
        phase_pce_weight: float = 0.1,
        phase_temperature: float = 0.02,
        phase_use_hann: bool = True,
        phase_demean: bool = True,
        phase_exclude_radius: int = 7,

        #------------Spectral---------------
        spectral_r_bins: int = None,    
        spectral_tau: float = 0.06,
        spectral_band_sigma: float = 0.15,
        spectral_w_peak: float = 1.0,
        spectral_w_band: float = 1.0,
        spectral_min_bin: int = 2,
        spectral_max_bin: int = None, 

        #-------------ACF-------------
        afc_r_bins: int = None,
        afc_tau: float = 0.08,
        afc_min_rel: float = 0.05,
        afc_max_rel: float = 0.6,
        afc_w: float = 1.0,

        #---------LOG----------------
        log_out_r: int = 128,
        log_out_t: int = 180,
        log_r_min: float = 0.02,
        log_w: float = 1.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.min_coverage = min_coverage

        # Параметры под маленький патч (64x64)
        r_bins = min(64, patch_size // 2)
        self.phase = PhaseCorrelationLoss(
            center_weight= phase_center_weight,
            pce_weight = phase_pce_weight,
            temperature = phase_temperature,
            use_hann = phase_use_hann,
            demean = phase_demean,
            exclude_radius = phase_exclude_radius,
        )
        self.spectral = SpectralPeriodLoss(
            r_bins = r_bins,
            tau = spectral_tau, 
            band_sigma = spectral_band_sigma,
            w_peak = spectral_w_peak, 
            w_band = spectral_w_band,
            min_bin = spectral_min_bin, 
            max_bin = r_bins - 1,
        )
        self.log = LogPolarAlignLoss(
            out_r = log_out_r, 
            out_t = log_out_t, 
            r_min = log_r_min, 
            w = log_w,
        )
        self.afc = ACFPeriodLoss(
            r_bins = r_bins,
            tau = afc_tau, 
            min_rel = afc_min_rel, 
            max_rel = afc_max_rel, 
            w = afc_w,
        )

        self.w_spectral = w_spectral
        self.w_afc = w_afc
        self.w_phase = w_phase
        self.w_log = w_log

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # 1) режем на патчи, фильтруем по покрытию маской
        pred_p, tgt_p, m_p = extract_masked_patches(
            pred, target, mask,
            patch_size=self.patch_size,
            stride=self.stride,
            min_coverage=self.min_coverage,
        )

        # 2) считаем лоссы на батче патчей (B_patches x C x p x p)
        L_spec = self.spectral(pred_p, tgt_p, m_p)
        L_afc  = self.afc(pred_p, tgt_p, m_p)
        L_phase= self.phase(pred_p, tgt_p, m_p)
        L_log = self.log(pred_p, tgt_p, m_p)

        # 3) комбинируем
        loss = (
            self.w_spectral * L_spec +
            self.w_afc      * L_afc  +
            self.w_phase    * L_phase +
            self.w_log      * L_log
        )
        print(f"local_loss:_{loss}")
        return loss
