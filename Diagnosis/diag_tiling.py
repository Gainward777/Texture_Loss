#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Диагностика тайлинга периодических текстур на изображениях.

Что делает:
  • Глобальный 2D-спектр (с/без окна Ханна), радиальный профиль и оценки f, 2f, 0.5f
  • ACF (автокорреляция) и период по первому пику
  • "Осевой крест" в FFT (axis energy ratio)
  • Карта локального периода (скользящее окно)
  • 1px shift-sensitivity (MSE при сдвиге по x/y)
  • Resize sanity (down→up с AA) и изменение фундаментала
  • JSON-отчёт + PNG-графики; есть функция вызова из тренировки

Зависимости: torch, torchvision, numpy, pillow, matplotlib

Примечание по FFT/окну:
  - 2D FFT берём из torch.fft (fft2/ifft2/fftshift) — стандартные API PyTorch.  :contentReference[oaicite:0]{index=0}
  - Ханново окно помогает подавить краевые/осевые артефакты при FFT.           :contentReference[oaicite:1]{index=1}
  - ACF считаем из PSD через теорему Винера–Хинчина.                            :contentReference[oaicite:2]{index=2}
  - Для sanity ресайза используйте антиалиас (torchvision Resize antialias).    :contentReference[oaicite:3]{index=3}
"""

from __future__ import annotations
import argparse, json, math, os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

EPS = 1e-8


# ---------------------------- utils: IO / tensors ----------------------------
def load_rgb(path: str) -> torch.Tensor:
    """PIL -> torch.FloatTensor CxHxW in [0,1]"""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)  # C,H,W
    return t


def load_mask(path: Optional[str], h: int, w: int) -> torch.Tensor:
    """Пороговая/мягкая маска в [0,1], 1xHxW"""
    if path is None:
        return torch.ones(1, h, w, dtype=torch.float32)
    m = Image.open(path).convert("L").resize((w, h), Image.BICUBIC)
    arr = (np.asarray(m).astype(np.float32) / 255.0)
    t = torch.from_numpy(arr)[None, ...]
    return t.clamp(0.0, 1.0)


def rgb_to_luma(t: torch.Tensor) -> torch.Tensor:
    """C,H,W -> 1,H,W (Rec.709)"""
    if t.shape[0] == 1:
        return t
    r, g, b = t[0], t[1], t[2]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return y.unsqueeze(0)


def hann2d(h: int, w: int, device=None, dtype=None) -> torch.Tensor:
    """2D Hann window (outer product). Снижает утечки/осевой крест на FFT. :contentReference[oaicite:4]{index=4}"""
    win_h = torch.hann_window(h, periodic=False, device=device, dtype=dtype)
    win_w = torch.hann_window(w, periodic=False, device=device, dtype=dtype)
    return (win_h[:, None] * win_w[None, :]).clamp_min(EPS)


# ---------------------------- FFT / profiles ---------------------------------
def fft_amp(y: torch.Tensor) -> torch.Tensor:
    """1xHxW -> HxW амплитуда FFT (центрованная)."""
    Y = torch.fft.fft2(y, norm="ortho")  # 2D DFT по последним двум осям :contentReference[oaicite:5]{index=5}
    A = torch.abs(Y).squeeze(0)          # HxW
    A = torch.fft.fftshift(A, dim=(-2, -1))
    return A


def power_spectrum(y: torch.Tensor) -> torch.Tensor:
    """1xHxW -> HxW PSD (амплитуда^2), центрованная."""
    A = fft_amp(y)
    return (A ** 2)


def autocorr_from_psd(psd: torch.Tensor) -> torch.Tensor:
    """H×W PSD -> H×W ACF (центр в середине). Теорема Винера–Хинчина. :contentReference[oaicite:6]{index=6}"""
    psd_shifted = torch.fft.ifftshift(psd, dim=(-2, -1))
    ac = torch.fft.ifft2(psd_shifted, norm="ortho").real
    ac = torch.fft.fftshift(ac, dim=(-2, -1))
    return ac


def radial_profile(image: torch.Tensor, n_bins: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Радиально-усреднённый профиль (mean по кольцам).
    Возврат: radii (px индексы 0..Rmax), values.
    """
    H, W = image.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = torch.meshgrid(torch.arange(H, device=image.device),
                            torch.arange(W, device=image.device), indexing="ij")
    r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = r.max()
    bins = torch.linspace(0, rmax + 1e-6, n_bins + 1, device=image.device)
    idx = torch.bucketize(r.reshape(-1), bins) - 1  # 0..n_bins-1
    idx = idx.clamp(0, n_bins - 1)
    vals = image.reshape(-1)
    prof = torch.zeros(n_bins, device=image.device)
    count = torch.zeros(n_bins, device=image.device)
    prof.index_add_(0, idx, vals)
    count.index_add_(0, idx, torch.ones_like(vals))
    prof = prof / (count + EPS)
    r_centers = 0.5 * (bins[:-1] + bins[1:])
    return r_centers, prof


def soft_peak(radii: torch.Tensor, profile: torch.Tensor, lo_bin: int, hi_bin: int, tau: float = 0.06):
    """Мягкий argmax по бинам [lo..hi]. Возврат: r_star (px), p (распределение)."""
    r = radii[lo_bin:hi_bin+1]
    v = profile[lo_bin:hi_bin+1]
    p = torch.softmax(v / max(tau, 1e-6), dim=0)
    r_star = (p * r).sum()
    return r_star, p


# ---------------------------- diagnostics ------------------------------------
@dataclass
class DiagConfig:
    out_dir: str
    radial_bins: int = 256
    min_bin: int = 2
    tau: float = 0.06
    local_win: int = 96
    local_stride: int = 32
    mask_thr: float = 0.3
    ring_width: int = 6  # для осевой метрики и колец f


def estimate_fundamental(y_luma: torch.Tensor, mask: torch.Tensor, cfg: DiagConfig, use_hann=True):
    """Оценка фундаментальной частоты по радиальному спектру."""
    y = y_luma * mask
    if use_hann:
        y = y * hann2d(*y.shape[-2:], device=y.device, dtype=y.dtype)
    A = fft_amp(y)
    radii, prof = radial_profile(torch.log1p(A), cfg.radial_bins)
    r_star, _ = soft_peak(radii, prof, cfg.min_bin, cfg.radial_bins - 2, tau=cfg.tau)
    return r_star.item(), (radii.cpu().numpy(), prof.cpu().numpy())


def energy_at_radius(A_log: torch.Tensor, r_star: float, ring_w: int, radii: torch.Tensor) -> Tuple[float, float, float]:
    """E(f), E(2f), E(0.5f) из лог-амплитуды (устойчивей к экспозиции)."""
    def window(center_r):
        w = torch.exp(- (radii - center_r)**2 / (2 * (ring_w ** 2)))
        w = w / (w.sum() + EPS)
        return w
    w_f  = window(r_star)
    w_2f = window(min(2 * r_star, radii[-1].item()))
    w_hf = window(max(0.5 * r_star, radii[1].item()))
    # построим профиль для текущей A_log
    rad, prof = radial_profile(A_log, len(radii))
    Ef  = float((w_f  * prof).sum().item())
    E2f = float((w_2f * prof).sum().item())
    Ehf = float((w_hf * prof).sum().item())
    return Ef, E2f, Ehf


def axis_energy_ratio(A: torch.Tensor, r_star: float, ring_w: int) -> float:
    """Осевой «крест»: энергия на осях в кольце вокруг f / общая энергия кольца."""
    H, W = A.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = torch.meshgrid(torch.arange(H, device=A.device), torch.arange(W, device=A.device), indexing="ij")
    r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    ring = torch.exp(- (r - r_star)**2 / (2 * (ring_w ** 2)))
    ring = ring / (ring.sum() + EPS)
    ax_mask = ((torch.abs(yy - cy) <= 1) | (torch.abs(xx - cx) <= 1)).float()
    num = (A * ring * ax_mask).sum()
    den = (A * ring).sum() + EPS
    return float((num / den).item())


def acf_period(y_luma: torch.Tensor, mask: torch.Tensor, cfg: DiagConfig) -> Tuple[float, np.ndarray]:
    """Период по первому пику ACF (px)."""
    y = y_luma * mask
    y = y * hann2d(*y.shape[-2:], device=y.device, dtype=y.dtype)
    psd = power_spectrum(y)
    acf = autocorr_from_psd(psd)
    radii, prof = radial_profile(acf.clamp_min(0), cfg.radial_bins)
    lo = int(0.03 * cfg.radial_bins)
    hi = int(0.6 * cfg.radial_bins)
    r1, _ = soft_peak(radii, prof, max(cfg.min_bin, lo), hi, tau=cfg.tau)
    return float(r1.item()), acf.cpu().numpy()


def local_period_map(y_luma: torch.Tensor, mask: torch.Tensor, cfg: DiagConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Карта локального периода (px) по скользящему окну."""
    C, H, W = 1, *y_luma.shape[-2:]
    win, stride = cfg.local_win, cfg.local_stride
    grid_y = list(range(0, max(1, H - win + 1), stride))
    grid_x = list(range(0, max(1, W - win + 1), stride))
    out = np.zeros((len(grid_y), len(grid_x)), np.float32)
    cov = np.zeros_like(out)
    for iy, y0 in enumerate(grid_y):
        for ix, x0 in enumerate(grid_x):
            patch = y_luma[:, y0:y0+win, x0:x0+win]
            m = mask[:, y0:y0+win, x0:x0+win]
            if m.mean().item() < cfg.mask_thr:
                out[iy, ix] = np.nan
                continue
            A = fft_amp((patch * m) * hann2d(win, win, device=patch.device, dtype=patch.dtype))
            radii, prof = radial_profile(torch.log1p(A), cfg.radial_bins)
            r_star, _ = soft_peak(radii, prof, cfg.min_bin, cfg.radial_bins - 2, tau=cfg.tau)
            r_val = float(r_star.item())
            period_px = float((max(win, win) / max(r_val, 1e-6)))
            out[iy, ix] = period_px
            cov[iy, ix] = float(m.mean().item())
    return out, cov


def roll_mse(y: torch.Tensor, mask: torch.Tensor, dx: int, dy: int) -> float:
    """MSE между изображением и его сдвинутой версией внутри усечённой маски."""
    y2 = torch.roll(y, shifts=(dy, dx), dims=(-2, -1))
    crop = 2
    m = mask.clone()
    if crop > 0:
        m[:, :crop, :] = 0; m[:, -crop:, :] = 0; m[:, :, :crop] = 0; m[:, :, -crop:] = 0
    num = ((y - y2) ** 2 * m).sum()
    den = m.sum() + EPS
    return float((num / den).item())


def resize_sanity(y: torch.Tensor, mask: torch.Tensor, scale=0.5) -> Tuple[float, float]:
    """down (AA) → up, MSE и сдвиг фундаментала (по радиусам). Антиалиас обязателен. :contentReference[oaicite:7]{index=7}"""
    C, H, W = y.shape
    y_img = (y.clamp(0, 1).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    y_pil = Image.fromarray(y_img.squeeze() if C==1 else y_img, mode="L" if C==1 else "RGB")
    down = y_pil.resize((int(W*scale), int(H*scale)), Image.BICUBIC)
    up   = down.resize((W, H), Image.BICUBIC)
    up_t = torch.from_numpy(np.asarray(up).astype(np.float32) / 255.0)
    if C == 1: up_t = up_t[None, ...]
    else:      up_t = up_t.permute(2,0,1)
    mse = float((((y - up_t) ** 2) * mask).sum() / (mask.sum() + EPS))
    r_pred, _ = estimate_fundamental(rgb_to_luma(y), mask, DiagConfig(""), use_hann=True)
    r_up, _   = estimate_fundamental(rgb_to_luma(up_t), mask, DiagConfig(""), use_hann=True)
    return mse, float(r_up - r_pred)


# ---------------------------- plotting ---------------------------------------
def imsave(path, arr, cmap="gray", vmin=None, vmax=None):
    plt.figure(figsize=(6,5)); plt.axis("off")
    plt.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


def plot_radial_profile(path, radii, prof, r_star):
    plt.figure(figsize=(6,4))
    plt.plot(radii, prof, label="log-amp radial")
    plt.axvline(r_star, color="r", linestyle="--", label="f*")
    plt.axvline(2*r_star, color="orange", linestyle=":", label="2f*")
    plt.axvline(0.5*r_star, color="green", linestyle=":", label="0.5f*")
    plt.xlabel("радиус частоты (бин FFT)"); plt.ylabel("усреднённая log-амплитуда")
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()


# ---------------------------- CLI --------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="путь к изображению с заменённой текстурой")
    ap.add_argument("--ref",  default=None, help="опционально: эталонный тайловый патч")
    ap.add_argument("--mask", default=None, help="опционально: PNG-маска стены (белое — стена)")
    ap.add_argument("--out",  required=True, help="папка для отчёта")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    pred = load_rgb(args.pred).to(torch.float32)
    C, H, W = pred.shape
    mask = load_mask(args.mask, H, W).to(torch.float32)
    y_pred = rgb_to_luma(pred).to(args.device)
    mask   = mask.to(args.device)

    # 1) Глобальный спектр с/без окна, радиальный профиль
    y0 = (y_pred * mask).to(args.device)
    A0 = fft_amp(y0)
    imsave(os.path.join(args.out, "fft_log_nohann.png"), torch.log1p(A0).cpu().numpy())
    y_h = y0 * hann2d(H, W, device=y0.device, dtype=y0.dtype)
    A_h = fft_amp(y_h)
    imsave(os.path.join(args.out, "fft_log_hann.png"), torch.log1p(A_h).cpu().numpy())

    cfg = DiagConfig(out_dir=args.out)
    r_star, (r_bins, prof) = estimate_fundamental(y_pred, mask, cfg, use_hann=True)
    plot_radial_profile(os.path.join(args.out, "radial_profile.png"), r_bins, prof, r_star)

    # метрики энергий на гармониках и осевой крест
    Ef, E2f, Ehf = energy_at_radius(torch.log1p(A_h), r_star, cfg.ring_width, torch.from_numpy(r_bins).to(A_h.device))
    axis_ratio = axis_energy_ratio(A_h, r_star, cfg.ring_width)

    # 2) ACF и период
    r_acf, acf = acf_period(y_pred, mask, cfg)
    imsave(os.path.join(args.out, "acf.png"), acf, cmap="magma")

    # 3) Локальная карта периода
    lp_map, lp_cov = local_period_map(y_pred, mask, cfg)
    imsave(os.path.join(args.out, "local_period_px.png"), lp_map, cmap="viridis")
    imsave(os.path.join(args.out, "local_mask_coverage.png"), lp_cov, cmap="plasma")

    # 4) Сдвиговая чувствительность и sanity ресайза
    mse_dx = roll_mse(y_pred, mask, dx=1, dy=0)
    mse_dy = roll_mse(y_pred, mask, dx=0, dy=1)
    mse_resize, dr = resize_sanity(y_pred, mask, scale=0.5)

    # 5) Если есть ref, оценим r_ref
    r_ref = None
    if args.ref:
        ref = load_rgb(args.ref).to(torch.float32)
        y_ref = rgb_to_luma(ref).to(args.device)
        r_ref, _ = estimate_fundamental(y_ref, torch.ones_like(mask), cfg, use_hann=True)

    # 6) Сводка
    summary = {
        "image": os.path.basename(args.pred),
        "H": H, "W": W,
        "fundamental_radius_bin": r_star,
        "acf_first_peak_radius_bin": r_acf,
        "axis_ratio_at_f": axis_ratio,
        "E2f_over_Ef": (E2f / (Ef + EPS)),
        "Ehalf_over_Ef": (Ehf / (Ef + EPS)),
        "shift_mse_dx1": mse_dx,
        "shift_mse_dy1": mse_dy,
        "resize_mse_downUp": mse_resize,
        "resize_delta_r": dr,
        "ref_radius_bin": r_ref,
        "notes": {
            "high_axis_ratio_hint": "высокие значения -> осевой «крест»/краевые эффекты FFT",
            "high_E2f_over_Ef_hint": "больше 1.0 -> модель «садится» на 2f (удвоение шага)",
            "high_shift_mse_hint": "сильная сдвиговая неинвариантность (alias/stride)",
        }
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # превью
    imsave(os.path.join(args.out, "pred.png"),
           (pred.permute(1,2,0).cpu().numpy() if C==3 else pred.squeeze().cpu().numpy()),
           cmap=None if C==3 else "gray")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


# ------------------ programmatic API: call from training ---------------------
def run_diagnostics_from_tensors(
    pred_bchw: torch.Tensor,              # B×C×H×W, float32 [0..1]
    mask_bchw: Optional[torch.Tensor],    # B×1×H×W, float32 [0..1] или None
    out_dir: str,                         # куда сохранять PNG/JSON
    ref_chw: Optional[torch.Tensor] = None,  # C×H×W (опц. референс-тайл)
    global_step: int = 0,
    writer=None,                          # опц. torch.utils.tensorboard.SummaryWriter
    tag_prefix: str = "train"
) -> Dict[str, float]:
    """
    Запуск диагностики «из обучения» по первому элементу батча.
    Возвращает словарь метрик (для логгера).
    """
    os.makedirs(out_dir, exist_ok=True)
    x = pred_bchw[0].detach().clamp(0,1).cpu()          # C×H×W
    C,H,W = x.shape
    m = (mask_bchw[0].detach().clamp(0,1).cpu()
         if mask_bchw is not None else torch.ones(1,H,W))
    y = rgb_to_luma(x)                                  # 1×H×W

    # FFT с/без окна, профиль, фундаментал
    A0 = fft_amp(y * m)
    imsave(os.path.join(out_dir, "fft_log_nohann.png"), torch.log1p(A0).numpy())
    yh = (y * m) * hann2d(H, W, device=y.device, dtype=y.dtype)
    Ah = fft_amp(yh)
    imsave(os.path.join(out_dir, "fft_log_hann.png"), torch.log1p(Ah).numpy())

    cfg = DiagConfig(out_dir=out_dir)
    r_star, (r_bins, prof) = estimate_fundamental(y, m, cfg, use_hann=True)
    plot_radial_profile(os.path.join(out_dir, "radial_profile.png"), r_bins, prof, r_star)

    Ef,E2f,Ehf = energy_at_radius(torch.log1p(Ah), r_star, cfg.ring_width, torch.from_numpy(r_bins))
    axis_ratio = axis_energy_ratio(Ah, r_star, cfg.ring_width)

    r_acf, acf = acf_period(y, m, cfg)
    imsave(os.path.join(out_dir, "acf.png"), acf, cmap="magma")

    lp_map, lp_cov = local_period_map(y, m, cfg)
    imsave(os.path.join(out_dir, "local_period_px.png"), lp_map, cmap="viridis")
    imsave(os.path.join(out_dir, "local_mask_coverage.png"), lp_cov, cmap="plasma")

    mse_dx = roll_mse(y, m, dx=1, dy=0)
    mse_dy = roll_mse(y, m, dx=0, dy=1)
    mse_resize, dr = resize_sanity(x, m)

    r_ref = None
    if ref_chw is not None:
        r_ref, _ = estimate_fundamental(rgb_to_luma(ref_chw.cpu()), torch.ones_like(m), cfg, use_hann=True)

    summary = {
        "fundamental_radius_bin": r_star,
        "acf_first_peak_radius_bin": r_acf,
        "axis_ratio_at_f": axis_ratio,
        "E2f_over_Ef": (E2f / (Ef + EPS)),
        "Ehalf_over_Ef": (Ehf / (Ef + EPS)),
        "shift_mse_dx1": mse_dx,
        "shift_mse_dy1": mse_dy,
        "resize_mse_downUp": mse_resize,
        "resize_delta_r": dr,
        "ref_radius_bin": r_ref,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if writer is not None:
        writer.add_scalars(f"{tag_prefix}/tiling", {
            "E2f_over_Ef": summary["E2f_over_Ef"],
            "Ehalf_over_Ef": summary["Ehalf_over_Ef"],
            "axis_ratio_at_f": summary["axis_ratio_at_f"],
            "shift_mse_dx1": summary["shift_mse_dx1"],
            "shift_mse_dy1": summary["shift_mse_dy1"],
            "resize_delta_r": summary["resize_delta_r"],
        }, global_step)
        writer.add_image(f"{tag_prefix}/fft_log_hann", torch.log1p(Ah)[None, ...], global_step, dataformats="CHW")
        writer.add_image(f"{tag_prefix}/local_period_px",
                         torch.from_numpy(lp_map)[None, None, ...], global_step, dataformats="NCHW")

    return summary


if __name__ == "__main__":
    main()
