# پژوهشگر شمارهٔ ۳ — CH4-METRICS-BACKFILL-01

> **مسیرهای معتبر پس از سبک‌سازی (۱۵ نوامبر ۲۰۲۶):** چهار ZIP و جدول‌های اصلی ۷۶ اجرای §۴.۱۲ در `code/experiments/standard_policy_comparison/results/` محفوظ‌اند. نتایج منحصربه‌فرد تاریخی و گزارش‌های مرحله‌ای R03 به `code/experiments/archive/benchmark_v1_legacy_evidence/` منتقل شده‌اند و Git blob اصلی آن‌ها تغییر نکرده است. اسکریپت‌های بازسازی از مسیر رسمی ZIP استفاده می‌کنند؛ گزارش‌های قدیمی زیر، وضعیت تاریخی زمان نگارش خود را ثبت می‌کنند و لزوماً مسیر فعلی فایل‌ها را نشان نمی‌دهند. راهنمای مسیرها در `code/experiments/archive/benchmark_v1_legacy_evidence/README_FA.md` آمده است.


**Deliverable branch:** `experiment/ch4-metrics-backfill-v2`  
**Immutable scientific reference:** `chapter4@de02fdda91e23db177de4429b3e1a9693c046d83`  
**Contract:** `code/experiments/COMMON_METRICS_CONTRACT.json` / `chapter4-common-evaluation-contract-v2`  
**Status:** `PARTIAL_WITH_REASONS` — 76/76 inventory records; **36/76** directly SHA-verified, center-only backfilled; no optimizer or walk replay.

این خروجی‌ها از داده‌های ذخیره‌شده ساخته شده‌اند؛ هیچ فایل تاریخی، فایل LaTeX، شاخه `chapter4` یا `main` تغییر نکرده است. تا تأیید ممیز مستقل، نتایج ۳۶تایی نباید «گزارش کامل v2 برای تمام ۷۶ اجرا» معرفی شوند.

| فایل | کاربرد |
|---|---|
| `coverage_manifest.json` | دقیقاً ۷۶ رکورد با ۵۵ شناسهٔ معیار، وضعیت و reason و SHA منبع؛ شمارش پوشش هر معیار |
| `coverage_manifest.csv` | همان ۷۶ رکورد در یک جدول wide با ۱۲۹ ستون، وضعیت و reason هر معیار |
| `coverage_summary.csv` | پنج شمارندهٔ وضعیت برای هر یک از ۵۵ metric |
| `verified_sources_36.json` | ۳۶ ماتریس مرکز ۳×۳ استخراج‌شده از دو ZIP با hash معتبر و دادهٔ خلاصهٔ خام |
| `center_only_metrics.csv` | ۳۶ ردیف T/BT/TNL/BTNL/I و ماتریس‌های JS و √JS و کمینه/میانگین فاصله |
| `discrepancy_report.md` | شرح تطبیق نسخه‌ها، عدم قطعیت، کنترل KL، tie و ماسک و گیت تحویل |
| `boundary_fixture_results.json` | نتیجهٔ واقعی fixture مستقل با NumPy/Torch؛ نه CI یا import از repo |
| `center_only_rebuild.py` | backfill **فقط از مرکز** با فراخوانی `code/policy_evaluator/metrics.py::center_metrics` و SHA guard؛ بدون آموزش/گشت |
| `test_backfill_boundaries.py` | regressionهای سبک برای اجرا در checkout؛ **در این مأموریت روی CI اجرا نشده** |
| `zip_extract.js` | ابزار فاقد وابستگی برای خواندن ZIP/DEFLATE و SHA-256 در مسیر دریافت شواهد |

## قرارداد عدم‌دسترسی

ZIPهای `extra_25_12runs.zip` (۱۲ اجرا) و `size_25_24runs_seeds20261103_20261104.zip` (۲۴ اجرا) مستقیماً خوانده شدند؛ SHA-256 کل ZIP، SHA اعضای JSON و مقادیر مانیفست، همگی کنترل شدند. دو ZIP `core_50_16runs.zip` (۱۶) و `size_25_24runs_seed20261102.zip` (۲۴) در این محیط ابزار، bytes قابل بازیابی نداشتند. **این وضعیت، نتیجهٔ علمی دربارهٔ خود داده‌ها نیست.** داده‌های scalar قدیمی از CSV آمده‌اند؛ status `measured` روی آن‌ها با `source_evidence=legacy_csv_only_unverified_raw_member` قابل تفکیک است. `R_KL` نسخهٔ اصلی و `NLL_KL_normalized_historical` به شکل و نسخهٔ پیشین نگهداری شده‌اند؛ از `η,ε` پایدار فصل سوم خبری نیست مگر در بررسی جداگانه.

## بازتولید و دستور اجرای کم‌هزینه

روی checkout شاخهٔ تحویل، از ریشهٔ پروژه:

```bash
python code/experiments/benchmark_v1/report_backfill_v2/center_only_rebuild.py
# خروجی پیش‌فرض فقط DRY_RUN است؛ برای بازگزارش محلیِ تنها مراکز و checksum ها:
python code/experiments/benchmark_v1/report_backfill_v2/center_only_rebuild.py --execute --out-dir /tmp/r03-center-only

# تست سبک؛ وابستگی: numpy و torch:
python -m unittest discover -s code/experiments/benchmark_v1/report_backfill_v2 -p 'test_backfill_boundaries.py' -v
```

`--execute` فقط ZIP و مرکز ذخیره‌شده را می‌خواند و هیچ داده‌ای را به موتور گشت یا optimizer نمی‌دهد. خروجی را خارج از آرشیو تاریخی می‌نویسد. این فرمان در جلسهٔ حاضر **اجرا نشده**؛ بنابراین در وضعیت فعلی فقط ۳۶ ردیف واقعاً backfill شده‌اند. checksumها و تعداد ۷۶ پیش از پذیرش مجموعهٔ کامل باید توسط QA مستقل دوباره کنترل شوند.

بستهٔ فعلی نیازی به Test ندارد. `node/node_class` بدون تابع آغاز نمونهٔ جدید، ۲۶ مقدار Validation را به‌درستی `not_applicable` می‌گذارد. در صورت نیاز به scoreها، margins، near ties یا exact KL per-sample، باید ابتدا مجوز گیت استنتاج صادر شود؛ **این مأموریت صفر بازپخش** داشته است.

**RESEARCHER-03**
