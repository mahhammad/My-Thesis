# E-CH4-03 | اعتبارسنجی همگرایی توزیع انتقال یک‌گامی به مرجع جامعه

**آزمایش رسمی بخش ۴.۳ پایان‌نامه؛ نسخه قابل ارجاع و مستقل از آرشیو قدیمی.**

برای توضیح سؤال علمی، پروتکل، تمام نتایج و محدودیت‌ها ابتدا [گزارش کامل فارسی](REPORT_FA.md) را بخوانید. نتایج قدیمی بدون بازاجرای محاسبات به این پوشه منتقل و با تست عددی ممیزی شده‌اند؛ [آرشیو تاریخی](../archive/q_population_validation/README_FA.md) حفظ شده است.

## راهنمای فایل‌ها

| مسیر | کاربرد |
|---|---|
| [compare.py](compare.py) | پیاده‌سازی و امکان اجرای دوباره مقایسه تحلیلی با برآورد محدود |
| [results/metrics.json](results/metrics.json) | منبع خام مرجع؛ ۱۲ تکرار در ۴ اندازه، مقدار ۳۰۰ اصلی، متادیتا |
| [results/replicates.csv](results/replicates.csv) | ۴۸ ردیف برای ممیزی میانگین و SD |
| [results/summary.csv](results/summary.csv) | ردیف‌های ۴ اندازه برای گزارش و نمودار |
| [results/per_feature.csv](results/per_feature.csv) | پنج ویژگی در هر اندازه مرجع |
| [tables/main_results.tex](tables/main_results.tex) | جدول جمع‌بندی با برچسب مورد استفاده در بخش ۴.۳ |
| [tables/per_feature_tv.tex](tables/per_feature_tv.tex) | جدول مقایسه ویژگی‌ها |
| [figures/error_convergence.tex](figures/error_convergence.tex) | نمودار میانگین TV، SD و √JSD با TikZ |
| [figures/per_feature_tv.tex](figures/per_feature_tv.tex) | نمودار پنج ویژگی با TikZ |
| [figures/error_convergence.svg](figures/error_convergence.svg) و [figures/per_feature_tv.svg](figures/per_feature_tv.svg) | نسخه‌های برداری قابل مشاهده مستقیم در GitHub؛ همان داده‌ها، با راهنمای انگلیسی |
| [results/provenance.json](results/provenance.json) | منشأ تاریخی، شناسه آزمایش، قرارداد seed و SHA بلا‌ب‌های Git |
| [test_compare.py](test_compare.py) و [test_report_assets.py](test_report_assets.py) | آزمون‌های سبک فرمول و تطبیق عددی فایل‌ها |
| [REPORT_FA.md](REPORT_FA.md) | **سند علمی مرجع** برای پژوهش، فصل چهارم و داوری |

## پروتکل حداقلی

پرس‌وجوهای ثابت Validation: ۹۰ نمونه و ۴۱۶ زوج دارای ویژگی مشاهده‌شده؛ مراجع ۳۰۰، ۱۰۰۰، ۳۰۰۰ و ۱۰٬۰۰۰تایی؛ ۱۲ تکرار؛ بدون سیاست، آموزش، گشت چندگامی یا دسترسی به Test. معیارهای TV و √JSD با مرجع تحلیلی همان کرنل تک‌ویژگی مقایسه می‌شوند. نتایج در [گزارش](REPORT_FA.md) و منبع [JSON](results/metrics.json) تشریح شده‌اند.

## راستی‌آزمایی بدون تکرار آزمایش

از ریشه مخزن:

`python -m unittest discover -s code/experiments/one_step_transition_population_convergence -p 'test_*.py' -v`

[اجرای CI این پوشه](https://github.com/mahhammad/My-Thesis/actions/workflows/chapter4-one-step-transition.yml) تغییرات داده، جداول و مختصات نمودار را کنترل می‌کند. این تست‌ها به داده Test رجوع نمی‌کنند. برای بازاجرای ۴۸ محاسبه اصلی فقط در صورت نیاز از `python code/experiments/one_step_transition_population_convergence/compare.py` استفاده کنید.

**قرارداد آزمایش‌های آینده:** [قالب گزارش مشترک فصل چهارم](../EXPERIMENT_REPORT_STANDARD_FA.md).
