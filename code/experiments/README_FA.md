# پیوست الکترونیکی فصل چهارم — آزمایش‌ها، معیارها و قابلیت بازتولید

این پوشه **پیوست غیرمتنی/الکترونیکی** فصل چهارم پایان‌نامه است. متن نظری معیارها در فصل سوم و قرارداد مشترک اجرای آن‌ها در بخش ۴.۲ فصل چهارم معرفی شده است. هر پوشه آزمایش شناسه و پروتکل مستقل دارد؛ جدول‌های پایان‌نامه فقط گزیده‌ای از داده‌ها هستند و سنجه‌های کامل/خام، محدودیت‌ها و آزمون‌ها در این مسیر نگهداری می‌شوند.

## اسناد مرجع

- [استاندارد گزارش آزمایش‌ها — نسخه ۲](EXPERIMENT_REPORT_STANDARD_FA.md): شرایط پذیرفته‌شدن نتیجه، مدیریت تاریخچه و روش ارجاع در LaTeX.
- [راهنمای سنجه‌های عمومی](COMMON_METRICS_FA.md): پیوند دقیق با فصل سوم؛ سه معیار R، معیارهای T/BT/TNL/BTNL/I، فاصله‌های بین مراکز، دقت و حاشیه، وضعیت داده‌های ناموجود.
- [قرارداد ماشین‌خوان مشترک](COMMON_METRICS_CONTRACT.json): فهرست نام‌های متریک، تجمیع متوازن، تعریف وضعیت‌ها و ساختار خروجی.
- [گزارشگر مستقل](benchmark_v1/report_snapshot.py): محاسبه خروجی کامل از بازنمایی‌ها و مراکز آموزش با توابع مرکزی فصل سوم در [metrics.py](../policy_evaluator/metrics.py).

## فهرست رسمی آزمایش‌های گزارش‌شده

| بخش | شناسه و عنوان | خروجی سنجه‌ها |
|---|---|---|
| **۴.۳** | [E-CH4-03 — اعتبارسنجی همگرایی انتقال یک‌گامی](one_step_transition_population_convergence/README_FA.md) | TV و جذر JSD برای Q؛ **معیارهای مرکز نامرتبط‌اند**. [پوشش](one_step_transition_population_convergence/results/coverage.json) |
| **۴.۴** | [E-CH4-04 — حساسیت سیاست یکنواخت](uniform_policy_hyperparameter_sensitivity/README_FA.md) | [گزارش کامل ۹ تنظیم](uniform_policy_hyperparameter_sensitivity/results/metrics_full.json) · [CSV مشترک](uniform_policy_hyperparameter_sensitivity/results/metrics_summary.csv) · [پوشش](uniform_policy_hyperparameter_sensitivity/results/coverage.json) |
| **۴.۵** | [E-CH4-05 — اختلاط احتمال و Guided بدون نویز](guided_informative_uniform_mixture/README_FA.md) | [گزارش کامل ۵ تنظیم](guided_informative_uniform_mixture/results/metrics_full.json) · [CSV مشترک](guided_informative_uniform_mixture/results/metrics_summary.csv) · [پوشش](guided_informative_uniform_mixture/results/coverage.json) |
| **۴.۶** | [E-CH4-06 — هدایت گشت به مراکز هدف](target_center_guidance_trajectory/README_FA.md) | [مراکز و معیارهای ۴۹ دور؛ ۳ وضعیت کامل](target_center_guidance_trajectory/results/metrics_full.json) · [پوشش و محدودیت](target_center_guidance_trajectory/results/coverage.json) |
| **۴.۷** | [E-CH4-07 — ارزیابی تجربی زیان حاشیه‌ای سیگموئیدی](sigmoid_margin_observational_evaluation/README_FA.md) | [گزارش سه checkpoint و خلاصه آموزش](sigmoid_margin_observational_evaluation/results/metrics_full.json) · [پوشش](sigmoid_margin_observational_evaluation/results/coverage.json) · [تحلیل تخصصی](sigmoid_margin_observational_evaluation/results/sigmoid_validation_summary.csv) |

## قرارداد دسترسی به هر آزمایش

در هر پوشهٔ رسمی، `README_FA.md` نقش نقشه فایل‌ها، `REPORT_FA.md` نقش گزارش پژوهشی، `results/` نقش داده و معیارهای قابل ممیزی، `tables/` و `figures/` نقش منابع جدول و شکل برداری برای پایان‌نامه و `test_*.py` نقش کنترل سازگاری را دارند.

در آزمایش‌های دارای مرکز، **مجموعه معیارهای عمومی کامل** بر اساس فصل سوم گزارش می‌شوند؛ این مجموعه فقط به $I$ و $R_{\mathrm{JS}}$ محدود نیست. مقادیر تفصیلی هر کلاس، ماتریس فاصله مراکز و NLL/حاشیه در فایل‌های JSON نگهداری می‌شوند؛ CSV برای مقایسه سریع حالات است.

**استثنای واقعی یا فقدان اندازه‌گیری** در فایل `results/coverage.json` صریحاً توضیح داده می‌شود: در ۴.۳ مرکز وجود ندارد، و در ۴.۶ بازنمایی تک‌نمونه‌ای برای همه ۴۹ دور ذخیره نشده و BA فقط در ۲۵ دور ثبت شده است. هیچ مقدار خالی، صفر مصنوعی یا نمونه‌گیری مجدد پنهان جایگزین این موارد نمی‌شود.

## بازتولید و تبار داده

سناریوی مصنوعی پایه در [تنظیمات مولد](../data_generator/config.json) تعریف شده است. هر آزمایش باید بذر، اندازه Train/Validation، نسخه گشت، سیاست آغاز، تعریف کلاس‌فرضی یا مشترک، وزن‌های کلاس و نحوه آزمون Test را ثبت کند. تا انتخاب نهایی سیاست، Test تاریخی برای تنظیمات یا تفسیر برنده استفاده نمی‌شود.

کد و نتایج قدیمی در [archive](archive) حفظ شده‌اند؛ انتقال فایل به پوشه رسمی همراه با نگه‌داشتن Git blob SHA یا منشأ صریح انجام می‌شود. آزمایش‌های ۴.۴ و ۴.۵ برای تکمیل سنجه‌های عمومی، یک **بازارزیابی سبک امید دقیق بدون آموزش** را گذرانده‌اند و با خروجی‌های قبلی منطبق شده‌اند: [CI مشترک](https://github.com/mahhammad/My-Thesis/actions/runs/42769091658). آزمایش ۴.۶ فقط بازتحلیل داده‌های ذخیره‌شده است و ۴۹ دور آن دوباره اجرا نشده‌اند.

## قرارداد افزودن آزمایش بعدی

۱. پیش از اجرا، سؤال و پروتکل را در گزارش مستقل تعریف کنید.  
۲. معیارهای مشترک فصل سوم را برای همه تنظیم‌های قابل‌محاسبه ثبت کنید؛ مقادیر غایب را با دلیل در `coverage.json` نشان دهید.  
۳. جدول/شکل از فایل نتیجه ساخته و با آزمون سبک کنترل شود.  
۴. تنها یافته‌های لازم و رفرنس فصل سوم وارد `tex/chapter4.tex` شود؛ گزارش مفصل و کد در این پیوست بماند.  
۵. وضعیت اجرا، صحت‌سنجی، یکپارچه‌سازی و تأیید کامپایل را جدا ثبت کنید.

آزمایش‌های **۴.۳ تا ۴.۷** با شناسه پایدار ثبت شده‌اند؛ پیش از هر آزمایش بعدی، یکپارچگی نام‌گذاری، قرارداد ارائه و پوشش معیارهای ناقص کنترل می‌شود.
