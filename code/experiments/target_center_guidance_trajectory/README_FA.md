# E-CH4-06 | هدایت گشت به مراکز هدف و تحلیل مسیر معیارهای بازنمایی

**قرارداد عمومی سنجش:** [فصل سوم و پیوست الکترونیکی](../COMMON_METRICS_FA.md). [گزارش جامع پوشش‌دار](results/metrics_full.json) · [وضعیت مقادیر موجود/ناموجود](results/coverage.json). هر ۴۹ دور هندسه مراکز و R_JS دارد؛ R_KL و R_√JS فقط برای دو دور Guided دارای بازنمایی خام و Uniform موجود است؛ ۲۵ BA تاریخی حفظ شده است.

**پوشه رسمی آزمایش ۴.۶.** این پوشه داده‌های واقعی و تأییدشدهٔ مطالعه تاریخی «هدایت مراکز هدف» را بدون تغییر و بدون اجرای مجدد مونت‌کارلو نگه می‌دارد. [گزارش علمی فارسی](REPORT_FA.md) پروتکل، محدودیت‌ها و نتایج را با همان استاندارد بخش‌های ۴.۳ تا ۴.۵ توضیح می‌دهد.

## نمای کلی شواهد

۴۹ دور از صفر تا ۴۸ ثبت شده‌اند؛ تمام ۴۹ دور معیارهای هندسی مرکز دارند. **BA اعتبارسنجی فقط در ۲۵ دور اندازه‌گیری شده** و برای دیگر دورها مقدار صفر/درون‌یابی به کار نرفته است. دور چهار بهترین BA در همین ۲۵ دور است و با Validation انتخاب شده است. سه وضعیت Uniform، Guided دور صفر و Guided دور چهار به‌طور تفصیلی تحلیل شده‌اند.

مقایسه Uniform و Guided از نظر پروتکل استنتاج یکسان نیست: Uniform یک بازنمایی مشترک و Guided یک بازنمایی جدا برای هر فرض کلاس دارد. در نتیجه این مطالعه یک **آزمایش مشاهده‌ای برای فهم هندسه و معیارهاست**، نه انتخاب برنده در یک رقابت هم‌پروتکل.

## مسیر فایل‌ها

| مسیر | نقش |
|---|---|
| [REPORT_FA.md](REPORT_FA.md) | **گزارش رسمی و تفسیر علمی** |
| [results/provenance.json](results/provenance.json) | شناسه و SHA تمام ۱۱ فایل خامِ کپی‌شدهٔ بدون تغییر از آرشیو |
| [results/history_00_48.json](results/history_00_48.json) | مراکز هدف و تحقق‌یافته، اطلاعات گشت و BAهای واقعاً ثبت‌شده |
| [results/trajectory_metrics.csv](results/trajectory_metrics.csv) | مسیر کامل $I$، $R_{\mathrm{JS}}$، فاصله هدف–واقعی و BAهای موجود |
| [results/center_geometry_49.json](results/center_geometry_49.json) | ماتریس‌های مراکز، معیارهای جدایی و هندسه همه ۴۹ دور |
| [results/three_cases.json](results/three_cases.json) | Uniform، Guided دور صفر و دور چهار با پروتکل و مراکزشان |
| [results/detailed_three_cases.json](results/detailed_three_cases.json) | NLL/تراکم/حاشیه و سایر معیارهای دقیق سه وضعیت |
| [results/representations_uniform.json](results/representations_uniform.json) | بازنمایی نمونه‌های Uniform |
| [results/representations_guided_round0.json](results/representations_guided_round0.json)، [results/representations_guided_round4.json](results/representations_guided_round4.json) | بازنمایی تک‌نمونه‌ای دو وضعیت Guided |
| [tables/three_cases.tex](tables/three_cases.tex) | جدول اصلی سه وضعیت و caveat درباره پروتکل |
| [tables/selected_rounds.tex](tables/selected_rounds.tex) | جدول منتخب دورها با BA خالی برای موارد اندازه‌گیری‌نشده |
| [figures/trajectory_geometry.tex](figures/trajectory_geometry.tex) / [SVG](figures/trajectory_geometry.svg) | روند $I$ و $R_{\mathrm{JS}}$ در ۴۹ دور |
| [figures/observed_validation_ba.tex](figures/observed_validation_ba.tex) / [SVG](figures/observed_validation_ba.svg) | **تنها ۲۵ نقطه BA واقعاً اندازه‌گیری‌شده** |
| [analyze.py](analyze.py)، [detailed_metrics.py](detailed_metrics.py) | اسکریپت‌های تحلیل از آرشیو، بدون تکرار گشت |
| [test_analyze.py](test_analyze.py)، [test_detailed_metrics.py](test_detailed_metrics.py)، [test_report_assets.py](test_report_assets.py) | تست‌های عددی و کنترل شواهد |

تمام ۱۱ فایل خام با **همان Git blob SHA** در [آرشیو اصلی](../archive/center_matching_comparison/REPORT_FA.md) نیز باقی مانده‌اند. ابزار بازاجرای مونت‌کارلوی تاریخی [reproduce.py](../archive/center_matching_comparison/reproduce.py) در آرشیو نگه داشته شده و در این مرحله استفاده نشده است.

## آزمایش بدون محاسبه سنگین

از ریشه مخزن:

`python -m unittest discover -v -s code/experiments/target_center_guidance_trajectory -p 'test_*.py'`

تست‌ها باید تداوم مراکز هدف، فرمول معیارهای JSD، گزینش دور چهار فقط از مقادیر اندازه‌گیری‌شده، ۴۹ دور/۲۵ BA و انطباق جدول و شکل با داده خام را بررسی کنند. نیازی به تولید داده، آموزش سیاست یا آزمون تاریخی ندارند.

[الگوی مشترک گزارش‌های فصل چهارم](../EXPERIMENT_REPORT_STANDARD_FA.md) برای آزمایش‌های بعدی نیز مبناست.
