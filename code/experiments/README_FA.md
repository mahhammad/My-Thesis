# پیوست الکترونیکی فصل چهارم — آزمایش‌ها، معیارها و قابلیت بازتولید

این پوشه **پیوست غیرمتنی/الکترونیکی** فصل چهارم پایان‌نامه است. متن نظری معیارها در فصل سوم و قرارداد مشترک اجرای آن‌ها در بخش ۴.۲ فصل چهارم معرفی شده است. هر پوشه آزمایش شناسه و پروتکل مستقل دارد؛ جدول‌های پایان‌نامه فقط گزیده‌ای از داده‌ها هستند و سنجه‌های کامل/خام، محدودیت‌ها و آزمون‌ها در این مسیر نگهداری می‌شوند.

## اسناد مرجع

- [استاندارد گزارش آزمایش‌ها — نسخه ۲](EXPERIMENT_REPORT_STANDARD_FA.md): شرایط پذیرفته‌شدن نتیجه، مدیریت تاریخچه و روش ارجاع در LaTeX.
- [راهنمای سنجه‌های عمومی](COMMON_METRICS_FA.md): پیوند دقیق با فصل سوم؛ سه معیار R، معیارهای T/BT/TNL/BTNL/I، فاصله‌های بین مراکز، دقت و حاشیه، وضعیت داده‌های ناموجود.
- [قرارداد ماشین‌خوان مشترک](COMMON_METRICS_CONTRACT.json): فهرست نام‌های متریک، تجمیع متوازن، تعریف وضعیت‌ها و ساختار خروجی.
- [گزارشگر مستقل](benchmark_v1/report_snapshot.py): محاسبه خروجی کامل از بازنمایی‌ها و مراکز آموزش با توابع مرکزی فصل سوم در [metrics.py](../policy_evaluator/metrics.py).

## نمایهٔ جزئی آزمایش‌های §۴.۳ تا §۴.۷ (فهرست کامل §۴.۳ تا §۴.۱۳ در ادامه)

| بخش | شناسه و عنوان | خروجی سنجه‌ها |
|---|---|---|
| **۴.۳** | [E-CH4-03 — اعتبارسنجی همگرایی انتقال یک‌گامی](q-validation/README_FA.md) | TV و جذر JSD برای Q؛ **معیارهای مرکز نامرتبط‌اند**. [پوشش](q-validation/results/coverage.json) |
| **۴.۴** | [E-CH4-04 — حساسیت سیاست یکنواخت](uniform-hyperparameter-sensitivity/README_FA.md) | [گزارش کامل ۹ تنظیم](uniform-hyperparameter-sensitivity/results/metrics_full.json) · [CSV مشترک](uniform-hyperparameter-sensitivity/results/metrics_summary.csv) · [پوشش](uniform-hyperparameter-sensitivity/results/coverage.json) |
| **۴.۵** | [E-CH4-05 — اختلاط احتمال و Guided بدون نویز](fixed-policy-evaluation/README_FA.md) | [گزارش کامل ۵ تنظیم](fixed-policy-evaluation/results/metrics_full.json) · [CSV مشترک](fixed-policy-evaluation/results/metrics_summary.csv) · [پوشش](fixed-policy-evaluation/results/coverage.json) |
| **۴.۶** | [E-CH4-06 — هدایت گشت به مراکز هدف](center-target-observation/README_FA.md) | [تمام سنجه‌های ۴۹ دور بازپخش هدف‌های ثابت](center-target-observation/results/metrics_full.json) · [۲۵ دقت تاریخی + ۲۴ مقدار تازه؛ ممیزی R_JS](center-target-observation/results/coverage.json) |
| **۴.۷** | [E-CH4-07 — ارزیابی تجربی زیان حاشیه‌ای سیگموئیدی](sigmoid-margin-evaluation/README_FA.md) | [تحلیل مشاهده‌ای و سه سیاست مرجع ۳۰۰تایی](sigmoid-margin-evaluation/results/metrics_full.json) · [۳۶ اجرای Benchmark با معیارهای کامل](sigmoid-margin-evaluation/results/benchmark_size_metrics_full.json) · [پوشش](sigmoid-margin-evaluation/results/coverage.json) |

## قرارداد دسترسی به هر آزمایش

در هر پوشهٔ رسمی، **فقط یک فایل Markdown به نام `README_FA.md`** هم نقش گزارش فنی پژوهشی و هم نقشهٔ دقیق فایل‌ها را دارد؛ `results/` نقش داده و معیارهای قابل ممیزی، `tables/` و `figures/` نقش منابع جدول و شکل برداری برای پایان‌نامه و `test_*.py` نقش کنترل سازگاری را دارند.

در آزمایش‌های دارای مرکز، **مجموعه معیارهای عمومی کامل** بر اساس فصل سوم گزارش می‌شوند؛ این مجموعه فقط به $I$ و $R_{\mathrm{JS}}$ محدود نیست. مقادیر تفصیلی هر کلاس، ماتریس فاصله مراکز و NLL/حاشیه در فایل‌های JSON نگهداری می‌شوند؛ CSV برای مقایسه سریع حالات است.

**پوشش و منشأ** در فایل `results/coverage.json` هر آزمایش مشخص می‌شود. در ۴.۳ اساساً مرکز تعریف نشده است. در ۴.۶ تاریخچه اولیه فقط ۲۵ مقدار BA و سه وضعیت با بازنمایی کامل داشت؛ اکنون بازپخش هدف‌های ثابت برای تمام ۴۹ دور معیارهای عمومی را تکمیل کرده و مقدارهای تازه را از تاریخچه جدا نگه می‌دارد. در ۴.۷ فقط مرجع کامل ۳۰۰تایی و اندازه‌های استاندارد Benchmark مبنای نتیجه‌اند؛ اجراهای ۷۲تایی و ۲۶اجرایی قدیمی آرشیو محسوب می‌شوند. هیچ صفر مصنوعی یا مقایسه ناهم‌بودجه جایگزین فقدان نمی‌شود.

## بازتولید و تبار داده

سناریوی مصنوعی پایه در [تنظیمات مولد](../data_generator/config.json) تعریف شده است. هر آزمایش باید بذر، اندازه Train/Validation، نسخه گشت، سیاست آغاز، تعریف کلاس‌فرضی یا مشترک، وزن‌های کلاس و نحوه آزمون Test را ثبت کند. تا انتخاب نهایی سیاست، Test تاریخی برای تنظیمات یا تفسیر برنده استفاده نمی‌شود.

کد و نتایج قدیمی پس از بازنشستگی `code/experiments/archive/` از طریق [راهنمای بازیابی و تگ قفل‌شده](../../research_journal/archive_recovery/README_FA.md) قابل بازیابی‌اند؛ فایل‌های مرجع لازم برای ممیزی در مسیرهای غیرآرشیوی باقی مانده‌اند. آزمایش‌های ۴.۴ و ۴.۵ برای تکمیل سنجه‌های عمومی، یک **بازارزیابی سبک امید دقیق بدون آموزش** را گذرانده‌اند و با خروجی‌های قبلی منطبق شده‌اند: [CI مشترک](https://github.com/mahhammad/My-Thesis/actions/runs/42769091658). آزمایش ۴.۶ با همان هدف‌های ذخیره‌شده و بذر تاریخی برای ۴۹ دور بازپخش شده است؛ این کار آموزش هدف‌های جدید یا بذر مستقل نیست. [CI بازپخش و انطباق مراکز](https://github.com/mahhammad/My-Thesis/actions/runs/42782841819). آزمایش ۴.۷ سه سیاست ذخیره‌شده ۳۰۰تایی را بدون آموزش مجدد بازسنجی و مطالعه اندازه را از ۳۶ اجرای هم‌بودجه Benchmark استخراج کرده است.

## قرارداد افزودن آزمایش بعدی

۱. پیش از اجرا، سؤال و پروتکل را در گزارش مستقل تعریف کنید.  
۲. معیارهای مشترک فصل سوم را برای همه تنظیم‌های قابل‌محاسبه ثبت کنید؛ مقادیر غایب را با دلیل در `coverage.json` نشان دهید.  
۳. جدول/شکل از فایل نتیجه ساخته و با آزمون سبک کنترل شود.  
۴. تنها یافته‌های لازم و رفرنس فصل سوم وارد `tex/chapter4.tex` شود؛ گزارش مفصل و کد در این پیوست بماند.  
۵. وضعیت اجرا، صحت‌سنجی، یکپارچه‌سازی و تأیید کامپایل را جدا ثبت کنید.

آزمایش‌های **۴.۳ تا ۴.۷** با شناسه پایدار ثبت شده‌اند؛ پیش از هر آزمایش بعدی، یکپارچگی نام‌گذاری، قرارداد ارائه و پوشش معیارهای ناقص کنترل می‌شود.


## نقشهٔ رسمی پوشه‌ها مطابق §۴.۳ تا §۴.۱۳

پوشه‌های زیر **۱۱ مطالعه مستقل** هستند؛ هیچ کد آزمایشگاهی یا ابزار کمکی نباید هم‌سطح آنها به‌عنوان آزمایش اضافی شمرده شود. موتورهای عمومی، حلگرهای تاریخی و زیرساخت محاسباتی در [benchmark_v1](benchmark_v1/README_FA.md) مستند شده‌اند. بستهٔ `standard_policy_comparison` نتایج آزمایش §۴.۱۲ است و با `benchmark_v1` مشترک یکی نیست.

| بخش | پوشهٔ رسمی |
|---|---|
| §4.3 | [`q-validation/`](q-validation/README_FA.md) |
| §4.4 | [`uniform-hyperparameter-sensitivity/`](uniform-hyperparameter-sensitivity/README_FA.md) |
| §4.5 | [`fixed-policy-evaluation/`](fixed-policy-evaluation/README_FA.md) |
| §4.6 | [`center-target-observation/`](center-target-observation/README_FA.md) |
| §4.7 | [`sigmoid-margin-evaluation/`](sigmoid-margin-evaluation/README_FA.md) |
| §4.8 | [`true-horizon-effect/`](true-horizon-effect/README_FA.md) |
| §4.9 | [`exact-matrix-evaluation/`](exact-matrix-evaluation/README_FA.md) |
| §4.10 | [`inverse_attainable_standard/`](inverse_attainable_standard/README_FA.md) |
| §4.11 | [`bayes_center_reachability_standard/`](bayes_center_reachability_standard/README_FA.md) |
| §4.12 | [`standard_policy_comparison/`](standard_policy_comparison/README_FA.md) |
| §4.13 | [`ch4_loss_controlled_comparison/`](ch4_loss_controlled_comparison/README_FA.md) |

طرحِ **اجرا‌نشده** §۴.۱۴ از فهرست آزمایش‌های انجام‌شده خارج و در [Research Journal](../../research_journal/ch4-14-synthetic-data-structure-sensitivity-plan.md) نگهداری می‌شود. فایل‌های خام و خروجی مطالعات رسمی منتقل یا بازنویسی نشده‌اند.
