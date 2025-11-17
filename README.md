# texture_losses

Функции потерь для стабилизации **масштаба тайлинга** (плитка/кирпич) при замене материала на изображении c LoRA для Flux Kontext.

> Полная HTML‑версия документации (со стилями) доступна как `texture_losses_docs.html` (положите в `docs/` и опубликуйте через GitHub Pages при необходимости).

---

## Содержание
- [Обзор](#обзор)
- [Установка и зависимости](#установка-и-зависимости)
- [Общие соглашения](#общие-соглашения)
- [API: Классы лоссов](#api-классы-лоссов)
  - [SpectralPeriodLoss](#spectralperiodloss)
  - [LogPolarAlignLoss](#logpolaralignloss)
  - [ScaleConsistencyLoss](#scaleconsistencyloss)
  - [ACFPeriodLoss](#acfperiodloss)
- [Типы данных и форматы — сводная таблица](#типы-данных-и-форматы--сводная-таблица)
- [Пример использования](#пример-использования)
- [Практические примечания](#практические-примечания)
- [Ссылки](#ссылки)

---

## Обзор

Модуль `texture_losses.py` содержит набор дифференцируемых функций потерь, которые адресно воздействуют на **масштаб (период)** периодической текстуры и минимально затрагивают её фактуру. Это достигается через сравнение признаков в частотной области (радиально‑усреднённые спектры, лог‑полярные карты) и через автокорреляцию.

**Состав:**
- `SpectralPeriodLoss` — попадание в основной частотный пик + узкополосное согласование формы спектра.
- `LogPolarAlignLoss` — выравнивание амплитудных спектров в лог‑полярных координатах (масштаб/угол → сдвиги).
- `ScaleConsistencyLoss` — консистентность масштаба между двумя стохастическими прогонами при одинаковых условиях.
- `ACFPeriodLoss` — сравнение периодов по первому пику радиального профиля автокорреляции (через теорему Винера–Хинчина).

---

## Установка и зависимости

- Python 3.9+
- PyTorch >= 2.0

```bash
pip install torch torchvision
```

Поместите файлы рядом в проекте:
```
texture_helpers.py
texture_losses.py
```

---

## Общие соглашения

- **Входы:**  
  `pred`, `target` — `B×C×H×W`, `C∈{1,3}`, `float32` (обычно `[0,1]`);  
  `mask` — `B×1×H×W`, `float32` в `[0,1]` (бинарная или мягкая).

- **Яркость:** если вход 3‑канальный, внутри лоссов берётся **luma** (яркостный канал).

- **Частотные операции:** используются `torch.fft.fft2`/`ifft2` и `fftshift`. Лог‑полярная выборка реализована через `torch.nn.functional.grid_sample`; масштабирование — `torch.nn.functional.interpolate`.

- **Устойчивость:** во всех логарифмах/делениях используется малое `EPS`.

- **Сложность:** доминирует 2D БПФ: примерно `O(B·H·W·log(HW))` на батч.

---

## API: Классы лоссов

### SpectralPeriodLoss

**Идея:** выравнивает основной частотный пик (период) и «подтягивает» форму спектра в узкой полосе вокруг целевого периода — остальной спектр почти не штрафуется.

```python
class SpectralPeriodLoss(torch.nn.Module):
    def __init__(
        self,
        r_bins: int = 256,
        tau: float = 0.06,
        band_sigma: float = 0.15,
        w_peak: float = 1.0,
        w_band: float = 1.0,
        min_bin: int = 2,
        max_bin: int | None = None
    ): ...

    def forward(
        self,
        pred:   torch.Tensor,   # B×{1|3}×H×W, float32
        target: torch.Tensor,   # B×{1|3}×H×W, float32
        mask:   torch.Tensor,   # B×1×H×W,    float32 in [0,1]
        center_override: torch.Tensor | None = None  # B, нормированная частота [0,1]
    ) -> torch.Tensor:          # scalar, float32
```

**Входы:**  
`pred`, `target` — изображения; внутри берётся luma.  
`mask` — область материала; используется для аподизации перед БПФ.  
`center_override` — **опционально**: центр узкой полосы как нормированная частота `[0,1]` (например из глубины/приора).

**Выход:** скалярный `torch.Tensor` (`float32`), средний по батчу.

---

### LogPolarAlignLoss

**Идея:** переводим амплитудные спектры в лог‑полярные карты (масштаб/угол → сдвиги) и минимизируем MSE между ними.

```python
class LogPolarAlignLoss(torch.nn.Module):
    def __init__(self, out_r: int = 128, out_t: int = 180, r_min: float = 1e-2, w: float = 1.0): ...

    def forward(
        self,
        pred:   torch.Tensor,   # B×{1|3}×H×W, float32
        target: torch.Tensor,   # B×{1|3}×H×W, float32
        mask:   torch.Tensor    # B×1×H×W,    float32
    ) -> torch.Tensor:          # scalar, float32
```

**Входы:** `pred`, `target` (берётся luma, затем FFT амплитуда и лог‑полярное отображение), `mask` — аподизация.  
**Выход:** скалярный `torch.Tensor` (`float32`), средний по батчу.

---

### ScaleConsistencyLoss

**Идея:** стабилизирует масштаб между **двумя** стохастическими прогонами (разные сиды/шумы) при одинаковых условиях — штрафует разницу основных частотных пиков (в лог‑шкале).

```python
class ScaleConsistencyLoss(torch.nn.Module):
    def __init__(self, r_bins: int = 256, tau: float = 0.06, min_bin: int = 2, max_bin: int | None = None, w: float = 1.0): ...

    def forward(
        self,
        pred1:  torch.Tensor,   # B×{1|3}×H×W, float32
        pred2:  torch.Tensor,   # B×{1|3}×H×W, float32
        mask:   torch.Tensor    # B×1×H×W,    float32
    ) -> torch.Tensor:          # scalar, float32
```

**Входы:** `pred1`, `pred2` — два предсказания при одинаковых условиях; `mask` — область материала.  
**Выход:** скалярный `torch.Tensor` (`float32`), средний по батчу.

---

### ACFPeriodLoss

**Идея:** период оценивается по первому значимому пику радиального профиля автокорреляции (ACF), которую получаем как обратное БПФ от спектра мощности (теорема Винера–Хинчина).

```python
class ACFPeriodLoss(torch.nn.Module):
    def __init__(self, r_bins: int = 256, tau: float = 0.08, min_rel: float = 0.03, max_rel: float = 0.6, w: float = 1.0): ...

    def forward(
        self,
        pred:   torch.Tensor,   # B×{1|3}×H×W, float32
        target: torch.Tensor,   # B×{1|3}×H×W, float32
        mask:   torch.Tensor    # B×1×H×W,    float32
    ) -> torch.Tensor:          # scalar, float32
```

**Входы:** `pred`, `target` — изображения; внутри вычисляются PSD и ACF. `mask` — аподизация.  
**Выход:** скалярный `torch.Tensor` (`float32`), средний по батчу.

---

## Типы данных и форматы — сводная таблица

| Объект/параметр | Тип | Форма | Диапазон/заметки |
|---|---|---|---|
| `pred`, `target` | `torch.Tensor` | `B×C×H×W`, `C∈{1,3}` | `float32` (рекоменд.), обычно `[0,1]`; устройство CPU/GPU согласовано |
| `mask` | `torch.Tensor` | `B×1×H×W` | `float32` в `[0,1]` (мягкая/бинарная); используется для аподизации |
| `center_override` (в `SpectralPeriodLoss.forward`) | `torch.Tensor` \| `None` | `B` | Нормированная частота **[0,1]** (радиальная), задаёт центр «узкой полосы» |
| Возвращаемое значение `forward(...)` | `torch.Tensor` | **Скаляр** | Средний по батчу лосс, `float32`, пригоден для `backward()` |

---

## Пример использования

```python
from texture_losses import (
    SpectralPeriodLoss, LogPolarAlignLoss,
    ScaleConsistencyLoss, ACFPeriodLoss
)
import torch
import torch.nn.functional as F

# pred/target/mask: torch.Tensor, формы B×C×H×W (C∈{1,3}) и B×1×H×W
spec = SpectralPeriodLoss(r_bins=256, w_peak=0.1, w_band=0.05)
lp   = LogPolarAlignLoss(out_r=128, out_t=180, r_min=1e-2, w=0.02)
sc   = ScaleConsistencyLoss(r_bins=256, w=0.05)
acf  = ACFPeriodLoss(r_bins=256, w=0.02)

L_rec  = F.mse_loss(pred * mask, target * mask)
L_spec = spec(pred, target, mask)       # scalar float32
L_lp   = lp(pred, target, mask)         # scalar float32
L_sc   = sc(pred1, pred2, mask)         # scalar float32 (два прогона)
L_acf  = acf(pred, target, mask)        # scalar float32

loss = L_rec + L_spec + L_lp + L_sc + L_acf
loss.backward()
```

---

## Практические примечания

- **Совпадение размеров:** `pred`, `target`, `mask` должны иметь одинаковые `B`, `H`, `W`.
- **DC‑компонента и хвост:** профили обрезают первые бины и далёкий хвост, чтобы исключить паразитные влияния.
- **AMP:** из‑за логарифмов и БПФ надёжнее держать вычисления в `float32` или использовать autocast c принудительным `float32` в критичных местах.
- **Интерполяция/сэмплинг:** используйте `nearest` для бинарных масок и `bilinear` для непрерывных карт.

---

## Ссылки

- PyTorch FFT: `torch.fft` — Fast Fourier Transforms  
- `torch.fft.fft2` / `torch.fft.ifft2` — документация PyTorch  
- `torch.nn.functional.grid_sample` — документация PyTorch  
- `torch.nn.functional.interpolate` — документация PyTorch  
- Теорема Винера–Хинчина (ACF ↔ PSD)
