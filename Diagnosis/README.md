Инструмент для поиска причин удвоения шага тайлинга и «сборки» артефактов по строкам/столбцам. Скрипт строит спектры (с/без окна Ханна), ACF, карту локального периода и считает метрики, которые помогают отличить: путаницу фундаментала с гармониками, осевые артефакты FFT, алиасинг при ресайзах и сетчатые артефакты ап/даун-семплинга. Основан на стандартных API PyTorch (torch.fft) и классических приёмах спектрального анализа. 
docs.pytorch.org

Что проверяет

Глобальный 2D-спектр (лог-амплитуда) с/без 2D-окна Ханна для снижения утечек и осевого «креста». Используются torch.hann_window и torch.fft.fft2/ifft2/fftshift. 
docs.pytorch.org
+1

ACF (автокорреляция) через PSD по теореме Винера–Хинчина — первый пик ACF даёт базовый период. 
comm.toronto.edu
+1

Отношения энергий гармоник E(2f)/E(f) и E(0.5f)/E(f) — индикатор «перескока» на 2f или ½f.

Axis energy ratio — доля энергии на горизонт/вертик. осях в кольце около f* (осевой «крест»).

Карта локального периода (скользящее окно) — визуализирует «столбики/строки» артефактов.

Shift-sensitivity (MSE при 1-px сдвиге) и Resize-sanity (down→up с AA) — маркеры сдвиговой неинвариантности и алиасинга; включение антиалиаса перед даунсэмплом — признанный способ стабилизации. 
Proceedings of Machine Learning Research
+1

Установка
pip install torch torchvision pillow numpy matplotlib


Документация torch.fft.fft2/ifft2 — для 2D БПФ. 
docs.pytorch.org
+1

Окно Ханна: torch.hann_window / torch.signal.windows.hann. 
docs.pytorch.org
+1

В torchvision.transforms.Resize включайте antialias=True для тензоров (в PIL bilinear/bicubic AA уже применяется). 
docs.pytorch.org

Запуск (CLI)
python diag_tiling.py \
  --pred path/to/replaced_wall.jpg \
  --mask path/to/wall_mask.png \    # опционально
  --ref  path/to/reference_tile.png \  # опционально (тайловый патч)
  --out  diag_report/


В папке появятся:

fft_log_nohann.png, fft_log_hann.png — глобальные спектры (лог).

radial_profile.png — радиальный профиль с отметками f*, 0.5f*, 2f*.

acf.png — автокорреляция.

local_period_px.png — карта локального периода.

summary.json — агрегированные метрики.

Вызов из кода обучения
import diag_tiling as dt
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/tiling_diag")
stats = dt.run_diagnostics_from_tensors(
    pred_bchw=pred.detach(),     # B×C×H×W, [0..1]
    mask_bchw=mask.detach(),     # B×1×H×W
    out_dir="diag_reports/e000_s0000000",
    ref_chw=None,                # или тайловый патч C×H×W
    global_step=global_step,
    writer=writer,               # опционально: логи в TensorBoard
    tag_prefix="train",
)
print(stats)

Как читать метрики (summary.json)
Ключ	Что означает	Интерпретация
fundamental_radius_bin	Оценка базовой частоты f* по радиальному профилю	Бин по радиусу FFT
acf_first_peak_radius_bin	Первый значимый пик ACF (через PSD)	Должен быть согласован с f* по теореме Винера–Хинчина; большие расхождения → подозрение на гармонику. 
comm.toronto.edu

E2f_over_Ef, Ehalf_over_Ef	Энергия в окнах вокруг 2f* и 0.5f* к энергии на f*	>1 → путаница фундаментала с гармониками
axis_ratio_at_f	Доля осевой энергии в кольце вокруг f*	Высоко → осевой «крест» (краевые/непериодические эффекты)
shift_mse_dx1/dy1	MSE при 1-px сдвиге	Большие значения → слабая сдвиговая устойчивость (помогает пред-даунсэмпловый low-pass / BlurPool). 
Proceedings of Machine Learning Research

resize_mse_downUp, resize_delta_r	Потеря и сдвиг f* после down→up (AA)	Рост → алиасинг/moire; проверяйте antialias=True в Resize. 
docs.pytorch.org

ref_radius_bin	f* эталонного патча	Удобно сравнить с fundamental_radius_bin
Частые причины и что делать

Checkerboard / сетчатые артефакты апсемплинга (ConvTranspose, sub-pixel): возникают из-за неравномерного перекрытия ядер. Надёжный практический фикс — resize→conv вместо deconv. 
Distill
+1

Сдвиговая неинвариантность и алиасинг от strided-ops/пуллинга: применяйте антиалиасный low-pass перед даунсэмплом (BlurPool/AA-CNN), что стабилизирует выходы к малым сдвигам. 
Proceedings of Machine Learning Research
+1

Краевые/осевые артефакты FFT: используйте окно Ханна перед спектральным анализом, чтобы уменьшить «крест» и утечки. 
docs.pytorch.org

Справочные ссылки

PyTorch FFT API (torch.fft): обзор и fft2/ifft2. 
docs.pytorch.org
+1

Окна Ханна в PyTorch. 
docs.pytorch.org
+1

Теорема Винера–Хинчина (ACF ↔ PSD). 
comm.toronto.edu
+1

Deconvolution & Checkerboard (Distill/Google Research). 
Distill
+1

Anti-aliased CNNs / BlurPool (ICML-2019 + проект). 
Proceedings of Machine Learning Research
+1

torchvision.transforms.Resize(antialias=True) — когда и как работает AA. 
docs.pytorch.org
