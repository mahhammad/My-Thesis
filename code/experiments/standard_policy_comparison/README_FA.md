# پیوست الکترونیکی §۴.۱۲ — مقایسه استاندارد خانواده‌های سیاست

**شناسه مطالعه:** \`E-CH4-12\` | **وضعیت:** \`EXECUTED / EVIDENCE_ARCHIVED\`  
**متن اصلی:** فصل چهارم، §۴.۱۲، برچسب \`sec:ch4-benchmark-v1\`.  
**کارکرد این پوشه:** بستهٔ مستقل شواهد همراه پایان‌نامه، بدون وابستگی لازم به نام یا مسیر پوشهٔ پژوهشی اولیه. فایل‌های خام و خلاصه‌ها بدون دستکاری محتوا از آرشیو مطالعه به این پوشه **کپی** شده‌اند؛ نسخه توسعه‌ای برای کار پژوهشی محفوظ است.

## بازآرایی شواهد R03

ابزار بازسازی و شواهد علمی مختص این مطالعه اکنون در [پوشهٔ R03](r03_center_recovery/README_FA.md) قرار دارند؛ [مدرک نگارش و حدود رأی ممیزی](r03_center_recovery/section412_editorial_evidence/README_FA.md) نیز حفظ شده است. خروجی‌های میانی و اسکریپت‌های یک‌بارمصرف از نسخهٔ فعال کنار گذاشته شده‌اند و در تگ `backup/pre-cleanup-2026-11-15` قابل بازیابی هستند. داده‌های رسمی پوشهٔ `results/` تغییر نکرده‌اند.

## مطالعه و شواهد

۷۶ اجرای ثبت‌شده به چهار بلوک تقسیم می‌شوند: ۱۶ اجرای اصلی ۵۰گامی، ۱۲ اجرای هدف جایگزین ۲۵گامی، ۲۴ اجرای اندازه مرجع ۲۵گامی در مولد پایه و ۲۴ اجرای دیگر ۲۵گامی با دو مولد مستقل. این ۷۶ اجرا **۷۶ تکرار مستقل مولد نیستند**. چهار خانواده \`shared\`، \`class\`، \`node\` و \`node_class\` ارزیابی شدند؛ در قرارداد موجود، Validation برای دو خانواده گره‌ای \`not_applicable\` است، نه صفر.

از ۷۶ مرکز ذخیره‌شده، **۱۱ شناسه سنجه مرکز × ۷۶ اجرا = ۸۳۶ مقدار مستقل از بازنمایی نمونه‌ای ممیزی و پذیرفته شده‌اند**. پذیرش این سنجه‌ها را به BA، NLL یا KL نمونه‌ای تعمیم ندهید؛ پوشش هر شناسه به صورت جداگانه در مانیفست ثبت شده است.

## راهنمای پرونده

| پرونده | استفاده |
|---|---|
| [PROTOCOL_FA.md](PROTOCOL_FA.md) | مدل، مولد، بودجه و محدودیت‌های استنتاج |
| [protocol/base_generator_config.json](protocol/base_generator_config.json) | تنظیمات مولد داده پایه |
| [protocol/training_config_v1.json](protocol/training_config_v1.json) | پیکربندی اولیه سیاست و آموزش |
| [results/master_runs_76.csv](results/master_runs_76.csv) | فهرست ۷۶ اجرا، family/objective/seed/steps، دقت‌ها، شناسه عضو JSON و SHA-256 |
| [results/center_only_metrics.csv](results/center_only_metrics.csv) | ۷۶ سطر مقادیر جدایی/هندسه، ماتریس‌های JS و √JS مراکز |
| [results/posthoc_score_backfill_76_compact.csv](results/posthoc_score_backfill_76_compact.csv) | NLL/Softplus پس‌محاسبه‌ای؛ سطح اعتبار متفاوت از center-only |
| [results/coverage_summary_76.csv](results/coverage_summary_76.csv) | وضعیت جمعی ۵۵ شناسه سنجه |
| [results/coverage_manifest_76.json](results/coverage_manifest_76.json) | وضعیت هر سنجه در هر اجرا و منشأ آن |
| [results/archive_preflight.json](results/archive_preflight.json) | گواه‌های منشأ و کنترل آرشیو |
| [results/recovery_audit.json](results/recovery_audit.json) | کنترل بازیابی |
| [results/center_only_rebuild_audit.json](results/center_only_rebuild_audit.json) | صحت بازسازی ۷۶ مجموعه مراکز |
| [results/ci_provenance.json](results/ci_provenance.json) | منشأ کنترل CI ثبت‌شده |

### آرشیو خام خروجی‌ها

| فایل در \`results/\` | تعداد اعضای اجرا | SHA-256 آرشیو |
|---|---:|---|
| \`core_50_16runs.zip\` | ۱۶ | \`9609eb96c207316fde26934cbef4668ba287a275d4485175d1da7e798a297040\` |
| \`extra_25_12runs.zip\` | ۱۲ | \`061a00923ae69ac2f8b923947bd9cdea9d4f3dba1d1771063bcb3ecffc0201e3\` |
| \`size_25_24runs_seed20261102.zip\` | ۲۴ | \`d570fb3f1dcab66476f027e1ade17b4e87c7ca16ff7fa6c76fda87b260c84e9f\` |
| \`size_25_24runs_seeds20261103_20261104.zip\` | ۲۴ | \`fdb862462e237e5094204ff17a5bc1a79a0f1d17f3a9b9d90ce967dba8b11fdd\` |

هر آرشیو حاوی فایل‌های JSON اجراها، پارامترهای سیاست، مراکز و خروجی‌های ذخیره‌شده است. در جدول \`master_runs_76.csv\` ستون \`file\` نام عضو JSON و ستون \`sha256\` هش همان عضو است. چهار ZIP برای بررسی شواهد اصلی کافی‌اند، ولی **ادعای اجرای مستقل تمام آموزش‌ها از ابتدا** از وجود این فایل‌ها حاصل نمی‌شود.

**قرارداد مهم:** \`R_KL\` تاریخی دارای clipping محاسباتی است و همان KL دقیقِ \`support-aware\` نیست. \`R_sqrt_JS\` میانگین جذر JSD نمونه‌هاست، نه جذر میانگین JSD. داده آزمون برای انتخاب \`τ\` و سیاست دوباره خوانده نشده است.

این بسته برای توزیع به‌عنوان پیوست الکترونیکی قابل انتخاب است. فایل‌های پژوهشی و کدهای توسعه لازم نیست همراه این بسته به خواننده داده شوند، مگر آنکه بازاجرای کامل آموزش جداگانه خواسته شود.
