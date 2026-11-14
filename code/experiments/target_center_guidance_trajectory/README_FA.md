# E-CH4-06 | هدایت گشت به مراکز هدف و تحلیل مسیر معیارهای بازنمایی

**تکمیل بازپخش معیارها (۱۰ نوامبر ۲۰۲۶):** هر ۴۹ هدف ثابت تاریخچه با همان ۱۰۰ گشت، بذر و سیاست هدایت‌شده بازپخش شد و تمام مراکز با تاریخچه تطبیق عددی داشتند. اکنون در results/metrics_full.json#replayed_fixed_targets هر ۴۹ دور دارای سه تراکم R/R_JS/R_sqrt_JS، پنج جدایی T/BT/TNL/BTNL/I، شاخص‌های حاشیه/دقت/NLL و هندسه کامل هستند. در ۴۴ دور پس از ادامه تاریخچه قدیمی، R_JS ذخیره‌شده میانگین نامتوازن نمونه‌ها بوده است؛ بازپخش مقدار **متوازن کلاس‌ها** را طبق فصل سوم مستقل محاسبه کرد. از ۴۹ دقت Validation، ۲۵ مقدار به‌صورت تاریخی ثبت شده بودند و ۲۴ مقدار در این بازپخش تازه محاسبه شدند؛ انتخاب قبلی دور چهار تغییر نکرده است. این یک بازپخشِ هدف ثابت است، نه چندبذری مستقل یا مقایسه هم‌پروتکل با Benchmark.

[تست موفق بازپخش ۴۹ دور](https://github.com/mahhammad/My-Thesis/actions/runs/42782841819) · [شکل متوازن معیارها](figures/trajectory_geometry_balanced.svg) · [نقاط دقت تاریخی و بازپخش](figures/validation_ba_replayed_49.svg).


**قرارداد عمومی سنجش:** [فصل سوم](../COMMON_METRICS_FA.md)؛ [گزارش کامل](results/metrics_full.json) · [پوشش و تمایز منشأ](results/coverage.json). همه ۴۹ دور در بازپخشِ هدف‌های ثابت، سه تراکم، پنج جدایی و دقت Train/Validation دارند؛ ۲۵ دقت اعتبارسنجی از تاریخچه اصلی و ۲۴ مورد صرفاً حاصل بازپخش‌اند. R_JS متوازن از نو محاسبه شده و مقدار تاریخیِ نامتوازن ۴۴ دور نیز برای ممیزی حفظ شده است.

**پوشه رسمی آزمایش ۴.۶.** این پوشه داده‌های مطالعه تاریخی را بدون تغییر نگه می‌دارد و یک بازپخش مجزای هدف‌های ثابت را نیز همراه کنترل تطبیق مراکز و تمایز مقدار تاریخی/جدید گزارش می‌کند. [گزارش علمی فارسی](REPORT_FA.md) پروتکل، محدودیت‌ها و نتایج را با همان استاندارد بخش‌های ۴.۳ تا ۴.۵ توضیح می‌دهد.

## نمای کلی شواهد

۴۹ دور از صفر تا ۴۸ ثبت شده‌اند؛ تمام ۴۹ دور معیارهای هندسی مرکز دارند. **BA اعتبارسنجی در تاریخچه اصلی فقط در ۲۵ دور اندازه‌گیری شده بود** و ۲۴ مقدار دیگر از بازپخش آمده است. هیچ مقداری صفر/درون‌یابی نشده است. دور چهار بهترین BA در همین ۲۵ دور است و با Validation انتخاب شده است. سه وضعیت Uniform، Guided دور صفر و Guided دور چهار به‌طور تفصیلی تحلیل شده‌اند.

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
| [tables/selected_rounds_balanced.tex](tables/selected_rounds_balanced.tex) | جدول چاپی با $R_{\mathrm{JS}}$ متوازن و BA تاریخی؛ دورهای فاقد اندازه‌گیری در تاریخچه اولیه مشخص شده‌اند |
| [figures/trajectory_geometry_balanced.tex](figures/trajectory_geometry_balanced.tex) / [SVG](figures/trajectory_geometry_balanced.svg) | روند $I$ و $R_{\mathrm{JS}}$ متوازن در ۴۹ دور بازپخش |
| [figures/validation_ba_replayed_49.tex](figures/validation_ba_replayed_49.tex) / [SVG](figures/validation_ba_replayed_49.svg) | ۲۵ مقدار BA تاریخی با نشان پر، ۲۴ مقدار بازپخش جدید با نشان توخالی |
| [analyze.py](analyze.py)، [detailed_metrics.py](detailed_metrics.py) | اسکریپت‌های تحلیل از آرشیو، بدون تکرار گشت |
| [test_analyze.py](test_analyze.py)، [test_detailed_metrics.py](test_detailed_metrics.py)، [test_report_assets.py](test_report_assets.py) | تست‌های عددی و کنترل شواهد |

تمام ۱۱ فایل خام با **همان Git blob SHA** در [آرشیو اصلی](../archive/center_matching_comparison/REPORT_FA.md) نیز باقی مانده‌اند. ابزار بازاجرای مونت‌کارلوی تاریخی [reproduce.py](../archive/center_matching_comparison/reproduce.py) در آرشیو نگه داشته شده و در این مرحله استفاده نشده است.

## آزمایش بدون محاسبه سنگین

از ریشه مخزن:

`python -m unittest discover -v -s code/experiments/target_center_guidance_trajectory -p 'test_*.py'`

تست‌های سبک تداوم مراکز هدف، فرمول‌های JSD، گزینش دور چهار از ۲۵ مقدار تاریخی و انطباق جدول‌ها و شکل‌های تاریخی را کنترل می‌کنند. آزمون تکمیلی `test_replay_presentation.py` نیز هر ۴۹ دورِ بازپخش‌شده، تجمیع متوازن $R_{\mathrm{JS}}$، ۴۴ اختلاف قدیمی و ۲۵/۲۴ نقطه شکل جدید را کنترل می‌کند؛ این آزمون‌ها گشت یا آموزش جدید اجرا نمی‌کنند.

[الگوی مشترک گزارش‌های فصل چهارم](../EXPERIMENT_REPORT_STANDARD_FA.md) برای آزمایش‌های بعدی نیز مبناست.
