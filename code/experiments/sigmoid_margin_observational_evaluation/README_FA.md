# E-CH4-07 — ارزیابی تجربی زیان حاشیه‌ای سیگموئیدی

**شناسه رسمی:** E-CH4-07؛ مرجع استاندارد: Benchmark v1؛ [گزارش](REPORT_FA.md)، [معیارها](results/metrics_full.json)، [پوشش](results/coverage.json).

## سه دسته شواهد (بدون مخلوط‌کردن پروتکل‌ها)

- **تحلیل تشخیصی:** سه بازنمایی ذخیره‌شده از E-CH4-06 × شش پهنای τ، فقط برای مطالعه حساسیت نسبت به حاشیه. Uniform و Guided استنتاج هم‌پروتکل ندارند و برنده مستقیم ندارند.
- **آموزش مستقیم روی مرجع کامل ۳۰۰تایی:** خانواده shared، سه τ، ۵۰ گام Adam. سه ماتریس نهایی ذخیره‌شده بدون آموزش دوباره بازسنجی شده‌اند؛ گزارش کامل سه R، پنج جدایی، حاشیه، دقت و NLL در results/metrics_full.json و تأیید آن در GitHub Actions ثبت است.
- **مطالعه استاندارد اندازه مرجع:** شش اندازه ۵۰/۱۰۰/۱۵۰/۲۰۰/۲۵۰/۳۰۰ × دو خانواده shared/class × سه بذر مستقل مولد 20261102/03/04 = **۳۶ اجرا** با بودجه یکسان ۲۵ گام. جدول اصلی tables/size_study.tex از results/benchmark_size_3seeds.csv و results/benchmark_size_aggregate.csv ساخته شده است. منبع خام benchmark_v1/results/master_runs_76.csv با شناسه SHA هر اجرا و فایل‌های ZIP Benchmark است.

## تأیید پوشش کامل مطالعه اندازه
در [بازسنجی موفق ۳۶ checkpoint](https://github.com/mahhammad/My-Thesis/actions/runs/42786885281)، تمام خروجی‌های سیاست ذخیره‌شده با SHA256 متناظر در ZIPهای Benchmark تطبیق داده شدند. اجرای دوباره فقط `forward` امید دقیق بود؛ نه آموزش و نه دسترسی به Test. [خروجی کامل v2 هر ۳۶ اجرا](results/benchmark_size_metrics_full.json) شامل سه تراکم، پنج جدایی، فاصله مراکز، دقت، حاشیه، NLL، ماتریس خطا و سنجه‌های هر کلاس برای Train و Validation است. بدین‌ترتیب محدودیت جدول تخت Benchmark، دیگر فقدان معیار در گزارش رسمی این مطالعه نیست.

## فایل‌ها و حدود اعتبار

README_FA.md نقشه پیوست، REPORT_FA.md گزارش پژوهشی، results/coverage.json دلیل هر فقدان، results/metrics_full.json بازسنجی مرجع ۳۰۰تایی، tables/ و figures/ منابع مستقیم فصل هستند. کدهای replay_saved_full300.py و replay_benchmark_size.py بازسنجی بدون آموزش را انجام می‌دهند.

**مطالعات غیرهم‌پروتکل صرفاً آرشیوی:** اجرای ۷۲تایی و مطالعه ۲۶ اجرای زیرمجموعه‌ای از یک مولد در archive/margin_training و فایل‌های results/size_study، tables/direct_training_72.tex، tables/size_study_legacy_26.tex باقی مانده‌اند؛ این‌ها دیگر منبع نتیجه یا جدول اصلی بخش ۴.۷ نیستند. گزارش پیشین به‌عنوان REPORT_LEGACY_FA.md حفظ شده است.

مقایسه آموزش ۵۰گامی روی ۳۰۰ نمونه با اندازه‌مطالعه ۲۵گامی هم‌بودجه نیست. از Test تاریخی استفاده نشده و کامپایل کامل LaTeX هنوز مستقل تأیید نشده است.
