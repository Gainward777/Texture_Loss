# Диагностика проблем с лоссами в Diffusion/LoRA-тренировке

## 1. Введение

Инструмент для поиска и подтверждения причин артефактов тайлинга (удвоение шага, «столбики/строки», alias/moire) при замене материалов на стенах. Скрипт строит спектральные и корреляционные карты, считает метрики и сохраняет понятный отчёт (PNG + JSON). Подходит и для одиночных изображений, и для периодического вызова прямо из кода обучения.

## 2. Что диагностируем

1. «Кучкование» артефактов строго по строкам/столбцам очень похоже на сетчатые артефакты апсемплинга (transposed-conv / sub-pixel), известные как checkerboard artifacts. Они возникают из-за неравномерных перекрытий ядер при апсемплинге; типичный фикс — заменить на resize-conv (Nearest/Bilinear → Conv) или подогнать кратность ядра и шага.
**Симптомы:** артефакты идут «по клетке», узор «срывается» вдоль чётных строк/столбцов.
2. Удвоение шага — классическая путаница фундаментальной частоты с гармониками (модель «садится» на 2f или ½f вместо f), усугубляемая утечкой спектра и краевыми эффектами FFT. Проверяется через ACF/радиальный спектр/цепстр.
**Симптомы:** «плитка в два раза длиннее/короче», часто зонами; на спектре — пик на 2f сильнее f.

4. Алиасинг/moire при ресайзе (даунсемплирование без низкочастотной фильтрации) рождает регулярные «усреднённые» ряды/колонки и «неправильные» периоды. Для CNN и диффузий это проявляется как сдвиговая неинвариантность из-за страйдов и пуллинга без антиалиаса (лечится blur-pool/антиалиасом перед даунсэмплом).
5. раевые эффекты в 2D-FFT (предположение периодичности кадра) дают осевой «крест» энергии, что подталкивает модель к осевым деформациям и столбикам/строкам. Лечится аподизацией (окна Ханна/Тьюки) или Periodic-plus-Smooth перед спектральными вычислениями.

   ***Пункты 1 и 3 - скорее всего чушь, т.к. чистый MSE дает совсем другие проблемы, но на всякий случай оставил, вдруг там тоже, что-то обнаружим.***

## 3. Методология диагностики

Диагностика включает:

### 3.1. Анализ лосса

-   Графики `train_loss`, `ema_loss`, `per-step loss smoothing`.
-   Выявление всплесков (\> 3σ относительно скользящего среднего).
-   Проверка корреляции лосса с конкретными батчами/образцами.

### 3.2. Анализ латентов

-   Вычисление статистики латентов до и после UNet.
-   Поиск периодических структур с помощью:
    -   FFT,
    -   автокорреляции,
    -   спектральной плотности.
-   Обнаружение вертикальных/горизонтальных паттернов.

### 3.3. Pixel-space диагностика

-   Рендер исходного изображения / результата.
-   Визуализация разницы (heatmap).
-   Выделение аномалий в 2D через фильтр Собеля + connected-component.

### 3.4. Dataset health checks

-   Обнаружение тайлинга в исходных текстурах (FFT + peak detection).
-   Проверка валидности масок.
-   Проверка перекоса распределений яркости/контраста.

## 4. Функциональность скрипта-диагноста

Скрипт:

-   Загружает логи обучения.
-   Подгружает сохранённые латенты (если доступны).
-   Принимает входные изображения для сравнения.
-   Генерирует отчёт:
    -   графики лосса,\
    -   спектральный анализ,\
    -   карты ошибок,\
    -   индикаторы вероятных причин проблемы.

## 5. Запуск диагностики

### 5.1. CLI запуск

    python diagnose_loss.py \
      --logdir /path/to/logs \
      --images in.png out.png \
      --latents /path/to/latents \
      --report out_report

### 5.2. Запуск внутри тренировки

Внутри цикла обучения:

``` python
from diagnose_loss import LossDiagnostics

diag = LossDiagnostics(config)
if step % config.diag_interval == 0:
    diag.log_step(step, loss, ema_loss, model, batch)
```

## 6. Параметры диагностики

  Параметр            Описание
  ------------------- ----------------------------------------
  `--fft-threshold`   чувствительность к периодическим пикам
  `--spike-sigma`     порог отклонений лосса
  `--heatmap-scale`   усиление уровня карты ошибок
  `--max-images`      число изображений в отчёте

## 7. Интерпретация результатов

### 7.1. Вертикальные/горизонтальные полосы

Причины: - дисбаланс батч-нормализации,\
- периодичность в латентах,\
- аугментации, создающие регулярный паттерн.

### 7.2. Удвоение тайлинга

Причины: - тайлинговые исходники в датасете,\
- ошибки подготовки масок,\
- избыточная регуляризация модели.

### 7.3. Всплески лосса

Причины: - испорченный батч,\
- слишком большой learning rate,\
- разрыв EMA.

## 8. Как расширить диагностику

-   Добавить анализ внимания (cross-attn maps).
-   Сравнение нескольких ckpt.
-   Автоматическое формирование pdf-отчёта.

## 9. Лицензия

| Key                                   | Meaning                                                                  | How to read it                                                                                                           |
| ------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| `fundamental_radius_bin`              | Estimated **f*** (radial bin in FFT) from the Hann-windowed log-spectrum | Baseline spatial frequency                                                                                               |
| `acf_first_peak_radius_bin`           | First significant ACF peak (via PSD→ACF)                                 | Should match **f***; divergence hints a harmonic mix-up (½f or 2f). Wiener–Khinchin grounds this check. ([Википедия][1]) |
| `E2f_over_Ef`, `Ehalf_over_Ef`        | Energy around **2f*** and **0.5f*** relative to **f***                   | > 1 ⇒ the model latched onto a harmonic (visual “tile size ×2” / “½”).                                                   |
| `axis_ratio_at_f`                     | Axis energy (horizontal/vertical) within the ring at **f***              | High ⇒ “axis cross” from boundary/periodicity mismatch; explains row/column banding.                                     |
| `shift_mse_dx1`, `shift_mse_dy1`      | MSE between prediction and its 1-px shift                                | High ⇒ shift non-invariance (aliasing). Use anti-alias downsampling / BlurPool. ([arXiv][2])                             |
| `resize_mse_downUp`, `resize_delta_r` | Damage and **f*** drift after down→up (AA)                               | Large ⇒ resize pipeline introduces moiré/aliasing; ensure antialiasing is enabled. ([docs.pytorch.org][3])               |
| `ref_radius_bin`                      | **f*** for your reference tile                                           | Compare with `fundamental_radius_bin` to judge scale accuracy                                                            |

[1]: https://en.wikipedia.org/wiki/Wiener%E2%80%93Khinchin_theorem?utm_source=chatgpt.com "Wiener–Khinchin theorem"
[2]: https://arxiv.org/abs/1904.11486?utm_source=chatgpt.com "Making Convolutional Networks Shift-Invariant Again"
[3]: https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Resize.html?utm_source=chatgpt.com "Resize — Torchvision 0.24 documentation"

