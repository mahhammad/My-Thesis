# نمایهٔ شواهد تجربی پایان‌نامه IMC
**CURRENT / CHAPTER 4**. این سند فهرست مطالعه است، نه جایگزین فایل raw و گزارش تفصیلی. [پیوست آزمایش‌ها](../code/experiments/README_FA.md) و [فصل چهارم](../tex/chapter4.tex) مراجع اصلی‌اند.

| بخش / موضوع | طراحی و مدرک | وضعیت و محدودیت استنباط |
|---|---|---|
| §۴.۱–۴.۲ داده | سه کلاس، پنج ویژگی، ماسک فقدان، Train/Validation/Test جدا؛ مبنا β=0.4، λ=0.8، H=12 | [A1 foundations](https://github.com/mahhammad/My-Thesis/issues/49#issuecomment-6792642221) پذیرفته؛ Test را برای انتخاب مدل مصرف نکن |
| §۴.۳ انتقال یک‌گامی | [مرجع جمعیت و نمونه](../code/experiments/q-validation/README_FA.md) | Q یک‌گامی؛ معیارهای مرکز چندگامی به‌طور خودکار قابل اعمال نیست |
| §۴.۴ سیاست یکنواخت | [حساسیت پارامترها](../code/experiments/uniform-hyperparameter-sensitivity/README_FA.md) | توصیفی و وابسته به تنظیم |
| §۴.۵ هدایت اولیه | [سیاست از اهداف و مخلوط یکنواخت](../code/experiments/guided_informative_uniform_mixture/README_FA.md)، [بازبینی MC vs exact](../code/experiments/guided_informative_uniform_mixture/README_FA.md) | تفاوت MC300 و exact مهم؛ با بودجه MC بزرگ‌تر نزدیک‌شدن مشاهده شد |
| §۴.۶ replay | [۴۹ دور هدایت](../code/experiments/target_center_guidance_trajectory/README_FA.md) | ۴۹ checkpoint مستقل‌داده نیست |
| §۴.۷ زیان سیگموئیدی | [گزارش](../code/experiments/sigmoid_margin_observational_evaluation/README_FA.md)، [Benchmark](../code/experiments/benchmark_v1/README_FA.md) | تشخیص ۹۰ Val و شش τ روی سه وضعیت = ۱۸ تشخیص، نه ۱۸ آموزش؛ رتبه وابسته τ و کاهش زیان ≠ افزایش BA |
| §۴.۸ عمق سقف‌دار | [گزارش عمق](../code/experiments/walk_depth_effect/README_FA.md)؛ [ژورنال](2026-11-10-E-CH4-09-depth-effect.md) | پاسخ به H مختص آزمایش، نه قضیه عمومی |
| §۴.۹ اعتبارسنجی محاسبات دقیق | [گزارش و شواهد E-CH4-08](../code/experiments/exact_matrix_walk_validation/README_FA.md) | محاسبهٔ دقیق، مقایسه با مونت‌کارلو و کران کوتاه‌سازی عمق؛ مستقل از مطالعهٔ سقف واقعی H در §۴.۸ |
| §۴.۱۰ وارون مراکز دست‌یافتنی | [گزارش و شواهد §۴.۱۰](../code/experiments/inverse_attainable_standard/README_FA.md)، [موتورهای محاسباتی مشترک](../code/experiments/benchmark_v1/README_FA.md) | **۱۰۸/۱۰۸ ثبت**؛ سه مولد مستقل، هدف‌ها از سیاست شاهد؛ B01-v2 با ۱۸ اجرای پذیرفته‌شده، raw سایر بلوک‌ها با برچسب تاریخی PROVISIONAL_PENDING_FINAL_QA؛ audit baseline ACCEPT_WITH_FIXES؛ موفقیت solver ≠ یکتایی یا برتری همگانی |
| §۴.۱۱ مراکز بیزی بیرونی | [REPORT](../code/experiments/bayes_center_reachability_standard/README_FA.md)، [manifest](../code/experiments/bayes_center_reachability_standard/results/provenance_manifest.json)، [certified summary](../code/experiments/bayes_center_reachability_standard/results/certified_summary.json) | **۱۰۳ واحد** (۵۴ bound، ۶ accuracy، ۳۶ solver، ۶ replay، ۱ probe)، سه seed مولد 20261102/03/04، n=150/300، شش گراف پایه در دامنهٔ گواه محصورسازی‌شده؛ ۴۸ حساسیت **نه** گواه مستقل |
| §۴.۱۲ ۷۶ استاندارد | [master_runs_76.csv](../code/experiments/standard_policy_comparison/results/master_runs_76.csv)، [مدرک نگارش](../code/experiments/standard_policy_comparison/README_FA.md)، [گزارش تاریخی R03](https://github.com/mahhammad/My-Thesis/blob/a8c9027460e538fa04b8aab2f762eb4ba4c31e3d/code/experiments/benchmark_v1/report_backfill_v2/CONSOLIDATION_R03_FA.md) | ۷۶ اجرا = ۱۶ اصلی + ۱۲ objective + ۴۸ اندازه/دیگر، شامل ۳۶ اجرا shared/class در سه seed داده و شش اندازه؛ **۷۶ مرکز و ۸۳۶ مقدار center-only** با ACCEPT_QA76_CENTER_ONLY؛ بقیهٔ قرارداد ۵۵متریکی blanket-accepted نیست |
| §۴.۱۳ مقایسهٔ زیان | [گزارش فنی E-CH4-13](../code/experiments/ch4_loss_controlled_comparison/README_FA.md)، [رأی مستقل](https://github.com/mahhammad/My-Thesis/issues/80#issuecomment-6892392493) | ۱۲ fit واقعی، سه seed مولد و شش جفت کنترل‌شده؛ `ACCEPT_PHASE2_SCIENTIFIC` و PR #79 ادغام‌شده؛ Test برای انتخاب استفاده نشده، ادعای برتری جهانی مجاز نیست |
| §۴.۱۴ طرح حساسیت داده | [طرح در Research Journal](ch4-14-synthetic-data-structure-sensitivity-plan.md) | فقط طرح ثبت‌شده؛ در نسخه فعلی فصل چهارم نتیجهٔ عددی برای آن گزارش نشده است |
| مخزن تاریخی | [benchmark_v1](../code/experiments/benchmark_v1/README_FA.md)، [راهنمای بازیابی آرشیو](archive_recovery/README_FA.md) | آزمایش‌های ۷۲نمونه‌ای قدیمی، گزارش‌های اولیه و ۹ بسته ZIP برای تبار؛ جایگزین پروتکل استاندارد نیستند |

## ممیزی نگهداشت شواهد پیش از ادغام

[مقایسهٔ مستقیم ۱۱ پوشهٔ آزمایش با تگ پیش از پاک‌سازی، هویت فایل‌های خام و راهنمای اجرایی](CH4_TAG_PRESERVATION_AUDIT_2026-11-15_FA.md) — ممیزی ساختاری و provenance؛ نه گواه جدید علمی.

## قرارداد مقایسه و شمارش
- تعداد اجرای solver، تعداد seed داده، تعداد مثال، تعداد checkpoint و تعداد سلول معیار را هرگز یکی نشمار. فقط سه seed مستقل در §۴.۱۰/§۴.۱۱ شناخته شده‌اند؛ ۷۶ اجرا به معنای ۷۶ مولد مستقل نیست.
- دادهٔ Train/Validation/Test در تصمیم‌های انتخاب مدل باید جدا بمانند؛ BA و NLL در هر split و هر قرارداد استنتاج باید با هم‌نوع خود مقایسه شوند. سیاست node/node_class اگر تابع شروع F ندارد، Validation آن **تعریف نشده** است، نه صفر.
- R_KL دقیق support-aware و R_KL^(legacy,clip) متفاوت‌اند. [قرارداد معیارها](../code/experiments/COMMON_METRICS_FA.md) و [قرارداد ماشین‌خوان](../code/experiments/COMMON_METRICS_CONTRACT.json) ملاک نام‌گذاری و وضعیت فیلدهاست.
- پذیرش ممیزی «شاهد raw دقیق و شناسه» را از «توصیف چاپ‌شده در LaTeX» جدا کن. وضعیت PASS/FAIL/NOT_RUN را از اجرای واقعی بگیر، نه از نثر پرونده.
- مسیر بازتولید: raw/manifest → reporter → تست/CI → داوری محدوده‌دار → انتخاب جدول/نمودار متن. نتایج خطادار/مردود را در آرشیو نگه دار و نشانی نسخه جایگزین بده.
