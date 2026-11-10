# E-CH4-04 | ارزیابی سیاست یکنواخت و تحلیل حساسیت ابرپارامترهای گشت

**قرارداد عمومی سنجش:** [فصل سوم و پیوست الکترونیکی](../COMMON_METRICS_FA.md). [خروجی کامل مشترک ۹ تنظیم](results/metrics_full.json) · [CSV همه معیارها](results/metrics_summary.csv) · [وضعیت پوشش](results/coverage.json). هر سه R، پنج معیار جدایی، ماتریس فاصله مراکز و معیارهای تصمیم موجودند؛ ۹ حالت امید دقیق سبک بدون آموزش با داده‌های پیشین انطباق داده شده‌اند.

**آزمایش دوم رسمی فصل چهارم — بخش ۴.۴.** این پوشه نسخه گزارش‌پذیر، مستند و قابل بررسی **۹ اجرای قدیمیِ امید دقیق با سیاست یکنواخت و بدون آموزش** است. برای آماده‌سازی نمودارها و جداول، اجرای محاسباتی تازه‌ای انجام نشده و Test تاریخی باز نشده است.

[گزارش علمی کامل](REPORT_FA.md) را برای طرح سؤال، فرمول‌ها، قرارداد Train/Validation، فهرست ۹ حالت، نتیجه و محدودیت‌ها بخوانید.

## فایل‌ها

| فایل | کاربرد |
|---|---|
| [REPORT_FA.md](REPORT_FA.md) | گزارش مستقل آزمایش و تفسیر با معیارهای پذیرش علمی |
| [results/uniform_runs.csv](results/uniform_runs.csv) | منبع **۹ ردیف کامل** از ۱۳ خروجی تأییدشده اصلی |
| [results/provenance.json](results/provenance.json) | منشأ Git SHA، بذر و قرارداد اجرا، شناسه snapshot و ممنوعیت Test |
| [results/plot_groups.csv](results/plot_groups.csv) | سه گروه تغییر $\beta,\lambda,H$؛ استفادهٔ مجدد از مبنا، بدون اجرای اضافی |
| [tables/main_results.tex](tables/main_results.tex) | جدول اصلی ۹ حالت برای بخش ۴.۴ |
| [tables/diagnostics.tex](tables/diagnostics.tex) | جدول کمکی شامل تراکم، حاشیه، Accuracy و BA |
| [figures/depth_sensitivity.tex](figures/depth_sensitivity.tex) | نمودار عمق مؤثر و سه پارامتر گشت |
| [figures/geometry_vs_depth.tex](figures/geometry_vs_depth.tex) | نمودار دو معیار جدایی/تراکم برحسب عمق مؤثر |
| [figures/balanced_accuracy.tex](figures/balanced_accuracy.tex) | نمودار آرشیوی Train/Validation BA برای ۹ تنظیم؛ به دلیل ثابت‌بودن دقت اعتبارسنجی در متن اصلی وارد نمی‌شود |
| [figures/depth_sensitivity.svg](figures/depth_sensitivity.svg)، [geometry_vs_depth.svg](figures/geometry_vs_depth.svg)، [balanced_accuracy.svg](figures/balanced_accuracy.svg) | نمودارهای قابل‌پیش‌نمایش مستقیم GitHub، با همان داده |
| [test_report_assets.py](test_report_assets.py) | کنترل تطبیق CSV اصلی، داده مشتق، جدول، نمودار، مرجع شماره فصل و عدم استفاده از Test |

## یافته در یک نگاه

در هر نه تنظیم، **BA اعتبارسنجی ۹۶٫۷۷٪** باقی ماند، اما عمق مؤثر از ۱ تا ۲٫۲۶۲۳، $I$ از ۰٫۰۰۹۸۵ تا ۰٫۰۳۲۴۲ و $R_{\mathrm{JS}}$ از ۰٫۰۰۱۱۸ تا ۰٫۰۰۳۹۵ تغییر کرد. بنابراین ثبات دقت متوازن در این تقسیم به معنای ثابت‌بودن هندسه نیست، اما برابری BA نیز یکسانی برچسب تک‌تک نمونه‌ها را ثابت نمی‌کند.

## روش استفاده و بازبینی

از ریشه مخزن:

`python -m unittest discover -v -s code/experiments/uniform_policy_hyperparameter_sensitivity -p 'test_*.py'`

برای مشاهده طرح ۹ اجرا بدون ارزیابی دوباره:

`python code/experiments/benchmark_v1/experiment_fixed_baselines.py --mode uniform`

برای تولید snapshot جدید فقط در صورت نیاز، از گزینه `--execute` در اسکریپت اصلی استفاده می‌شود. دادهٔ Test تاریخی در هیچ‌کدام از کنترل‌های گزارش باز نمی‌شود.

**منشأ اجرا:** [GitHub Actions #42649911898](https://github.com/mahhammad/My-Thesis/actions/runs/42649911898)؛ هر ۹ snapshot در آن run بخشی از ۱۳ snapshot با round-trip مستقل موفق بودند. **تست نسخه گزارشی:** [CI #42715068662](https://github.com/mahhammad/My-Thesis/actions/runs/42715068662) با ۱۰ کنترل داده و نمودار و ۳ کنترل ایستای LaTeX، همگی موفق. این پوشه مستقل از گزارش سیاست هدایت‌شده است که به آزمایش بعدی تعلق دارد.

[استاندارد مشترک گزارش فصل چهارم](../EXPERIMENT_REPORT_STANDARD_FA.md) · [گزارش مبنای قبلی ۴.۳](../one_step_transition_population_convergence/REPORT_FA.md).
