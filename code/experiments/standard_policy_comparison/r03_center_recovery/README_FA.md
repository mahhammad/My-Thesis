# بازیابی هندسهٔ مراکز — پژوهشگر R03 / E-CH4-12

این زیرپوشه **ویژهٔ مطالعهٔ §۴.۱۲ (مقایسهٔ استاندارد خانواده‌های سیاست)** است و ابزار عمومی تمام آزمایش‌ها محسوب نمی‌شود.

## منابع معتبر

- شواهد رسمی: [۷۶ ردیف هندسهٔ مراکز](../results/center_only_metrics.csv)، [مانیفست ۷۶ اجرا](../results/coverage_manifest_76.json) و [ممیزی بازیابی](../results/recovery_audit.json).
- رأی مستقل [`ACCEPT_QA76_CENTER_ONLY`](https://github.com/mahhammad/My-Thesis/issues/49#issuecomment-6796097800) فقط **۷۶ مرکز و ۸۳۶ مقدار مرکز-محور** را پوشش می‌دهد؛ این حکم به ۴۴ شناسهٔ دیگر معیار تعمیم نمی‌یابد.
- [مدرک نگارش §۴.۱۲](section412_editorial_evidence/README_FA.md) و [گزارش محدودیت‌ها و اختلاف‌های R03](discrepancy_report.md) با Git blob اصلی و بدون تغییر بایت منتقل شده‌اند.

## ابزارهای بازتولید

- `center_only_rebuild.py`: بازسازی فقط معیار هندسهٔ مراکز از چهار ZIP رسمی، با کنترل SHA-256 آرشیو و اعضا؛ بدون آموزش، بازپخش گشت و دسترسی Test؛ **پیش‌فرض dry-run**.
- `test_backfill_boundaries.py`: آزمون مرزهای KL، softmax ماسک‌شده، tie و هندسهٔ مرکز.
- `boundary_fixture_results.json`: شاهد تاریخی fixture مستقل؛ به‌معنای موفقیت CI یا ممیزی کامل نیست.

از ریشهٔ مخزن:
```bash
python code/experiments/standard_policy_comparison/r03_center_recovery/center_only_rebuild.py
python code/experiments/standard_policy_comparison/r03_center_recovery/center_only_rebuild.py --execute --out-dir /tmp/ch4-r03-centers
python -m unittest discover -s code/experiments/standard_policy_comparison/r03_center_recovery -p 'test_backfill_boundaries.py' -v
```

## پرونده‌های تاریخی بازنشسته از checkout

جدول و مانیفست اولیهٔ **۳۶/۷۶** شامل `center_only_metrics.csv`، `coverage_manifest.csv/json`، `coverage_summary.csv` و `verified_sources_36.json` دیگر منبع جاری نیستند. هر ۳۶ اجرا و تمام ۲۵ ستون جدول اولیه با جدول رسمی ۷۶تایی مطابقت دارند (بیشینهٔ اختلاف عددی حدود `8.61e-16`، بدون مغایرت فراتر از `1e-10`). این تطبیق **فقط داده‌های هندسهٔ مراکز** را پوشش می‌دهد.

اسکریپت‌های مختص اجرای یک‌بارهٔ ترمیم `run_recovery.py`، `preflight_recovery.py`، `verify_recovery.py`، `budget_guard.py`، ابزار `zip_extract.js` و گزارش اداری `CONSOLIDATION_R03_FA.md` نیز از نسخهٔ فعال کنار رفته‌اند.

**بازیابی نسخهٔ اولیه:** [تگ قفل‌شدهٔ پیش از پاک‌سازی](https://github.com/mahhammad/My-Thesis/tree/backup/pre-cleanup-2026-11-15/code/experiments/benchmark_v1/report_backfill_v2). فایل‌های رسمی `results/`، متن فصل‌ها و نتایج علمی تغییر نکرده‌اند. مسیرهای قدیمی در اسناد پژوهشی تاریخی، ارجاع منشأ بوده‌اند و نباید به‌عنوان مسیر اجرایی فعال تفسیر شوند.
