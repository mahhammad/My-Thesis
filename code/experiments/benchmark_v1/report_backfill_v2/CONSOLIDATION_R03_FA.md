# RESEARCHER-03 — CH4-R03-CONSOLIDATE-01

**ماموریت:** Git-only consolidation برای شاخه دائمی `researcher/03-chapter4`.  
**وضعیت تحویل:** `READY_FOR_INDEPENDENT_TREE_QA / OLD_BRANCHES_NOT_DELETED`.  
**Issue رسمی:** https://github.com/mahhammad/My-Thesis/issues/57#issuecomment-6829908585  
**تاریخ کنترل:** ۲۰۲۶-۱۱-۱۲؛ بدون اجرای عددی، تغییر علم، rewrite/merge یا حذف ref.

## ۱) تثبیت تاریخچه و انتخاب پایه

| ترتیب | شاخه قدیمی و SHA قفل‌شده | نتیجه `compare` از مبدأ قبلی | وضعیت حفاظت |
|---|---|---|---|
| Backfill | `experiment/ch4-metrics-backfill-v2@0a552e6e31999f0a9aeab499f62d3b54313c2e1c` | ۵۶۰ Git blob در snapshot مبدأ | **EXACT** در P3 و شاخه جدید، کل commit ancestor |
| P1 | `experiment/ch4-metrics-p1-provisional-r03@d36ec6dc522e4e6d5f7ee4134f23c71f97b2aff3` | ۵ commit جلوتر از Backfill، `behind=0`، بدون حذف/تغییر blobهای قبلی؛ ۵ فایل افزودنی | **EXACT** در P3 و شاخه جدید، کل commit ancestor |
| P3 | `experiment/ch4-metrics-p3-publication-gates-r03@059cefd67e349490614010780abc0df6d58d5d94` | ۴ commit جلوتر از P1، `behind=0`؛ ۴ فایل افزودنی | **EXACT**، پایه ساخت مستقیم شاخه جدید |

Git commit `P3` دارای tree واقعی `5eb9a36ed21b94e35ae97191f107190dc08227da` است. قبل از افزودن این سند، HEAD و tree شاخه دائمی **عیناً همان P3** بودند. هر سه شاخه قدیمی هنوز موجود و بدون هیچ rewrite هستند. هیچ merge یا cherry-pick لازم نبود؛ `P1` و `Backfill` اجداد مستقیم `P3` و در نتیجه شاخه دائمی‌اند.

در تطبیق recursive Git trees: `Backfill=560` فایل، `P1=565` فایل، `P3=569` فایل؛ **صفر فایل ازدست‌رفته و صفر Git blob جایگزین‌شده** در هر مرحله، و فقط ۹ فایل جدید با مسیرهای مستقل. فایل کنونی تنها افزودهٔ مربوط به ثبت تجمیع است؛ هیچ یک از ۵۶۹ Git blob اصلی تغییر نکرده است.

## ۲) تطبیق کامل فایل‌ها / Git blob SHAها

تمام فایل‌های مرحله Backfill، از جمله `README_FA.md`، `coverage_manifest.{csv,json}`، `discrepancy_report.md`، `boundary_fixture_results.json`، `verified_sources_36.json`، scriptهای `preflight/recovery/verify`، خروجی‌های نسخه `full_76_candidate/`، `center_only_metrics.csv` و raw ZIPهای `results/`، با **همان SHA Git blob** در P3 باقی مانده‌اند.

فایل‌های افزودهٔ P1 و P3 بدون تغییر بایت در شاخه دائمی:

| مرحله | فایل (نسبت به `code/experiments/benchmark_v1/report_backfill_v2/`) | Git blob SHA | وضعیت |
|---|---|---|---|
| P1 | `p1_provisional/REPORT_FA.md` | `897ade1b368f5586cab9ea3983da33f5a6a4b743` | **EXACT** |
| P1 | `p1_provisional/coverage_76x55_provisional.csv` | `41e2d2f3252fd5b3145c9215f6aee8d225ab9dea` | **EXACT** |
| P1 | `p1_provisional/coverage_summary_55_provisional.csv` | `9f1b9ccecefb6d1cc766ece644cd77507aa29b93` | **EXACT** |
| P1 | `p1_provisional/metric_rules_55.json` | `3a41e325088c55a6112bb4c4e80df286637d899e` | **EXACT** |
| P1 | `p1_provisional/raw_aggregate_audit_76.json` | `d6d83e9937b58b68aa40cecc3994bf2542595a79` | **EXACT** |
| P3 | `p3_publication_gates/CH4_DRAFT_FA.md` | `5c966f3f9aeca810ee4277c99ec4024833c159a9` | **EXACT** |
| P3 | `p3_publication_gates/CLAIM_GATE_MATRIX_55x4.csv` | `58ed141562bc07b0549d5e1f0d28ed83c1f4232a` | **EXACT** |
| P3 | `p3_publication_gates/P2_MINIMAL_DECISION.md` | `8ec5c5b9042b00c7eb26c25366a9220728b3d98a` | **EXACT** |
| P3 | `p3_publication_gates/REPORT_FA.md` | `15c2e8872abd2cfd4405e599deda6d6e1a97a787` | **EXACT** |

برخی فایل‌های کلیدی Backfill با Git blob قفل‌شده:
- `full_76_candidate/center_only_metrics.csv@4e7ab942ea6a825639829b868e41ed50d654d14d`
- `full_76_candidate/coverage_manifest_76.csv@8135e30da33b7bd71c8681dc4a3569643a941667`
- `full_76_candidate/coverage_manifest_76.json@4af9372e27b7aaefe17fbe52e0aa68e5e4beea10`
- `full_76_candidate/archive_preflight.json@f4ea3635bfc90969e7daab8293f46d90568b26cf`
- `verified_sources_36.json@fe8ec364f9f2e9a00b3871234b177bda58dd512b`
- `results/master_runs_76.csv@bcc2d7f4a4f39301973eb53c465ff05a6b4d0f21`

**الزام حفاظت خام:** چهار ZIP اصلی `results/core_50_16runs.zip`، `extra_25_12runs.zip`، `size_25_24runs_seed20261102.zip` و `size_25_24runs_seeds20261103_20261104.zip` و دو ZIP اضافی `inverse_50_4families.zip` و `fixed_center_recheck_7runs.zip` با همان Git blob در Backfill/P1/P3/شاخه دائمی وجود دارند. در این مأموریت فقط Git blobها تطبیق داده شدند؛ SHA256 بایت ZIP از نو محاسبه نشد. SHA256 فایل‌های چهار آرشیو ۷۶ اجرا در `archive_preflight.json` محفوظ است.

## ۳) پوشش و اعتبار شواهد

- چهار آرشیو اصلی به‌ترتیب **۱۶، ۱۲، ۲۴ و ۲۴ عضو JSON** دارند (جمع **۷۶**). `run_id` همهٔ ۷۶ عضو منحصربه‌فرد است و با `master_runs_76.csv`، `center_only_metrics.csv`، `full_76_candidate/coverage_manifest_76.csv`، `p1_provisional/coverage_76x55_provisional.csv` و `p1_provisional/raw_aggregate_audit_76.json` یک‌به‌یک متصل است؛ **صفر عدم‌انطباق member SHA** در crosscheck منابع ذخیره‌شده.
- کاتالوگ P1 دقیقاً **۷۶×۵۵=۴۱۸۰** سلول وضعیت دارد: `measured=836`، `recomputed_from_raw=882` (فقط arithmetic over SAVED aggregate)، `historical_unverified=1286`، `not_measured=630`، `not_applicable=546` و `undefined_or_infinite=0` به‌عنوان **وضعیت گزارش**. این اعداد نه ۴۱۸۰ metric مستقل‌اند و نه ادعای finiteness.
- کاتالوگ P3 دقیقاً **۲۲۰ سطر = ۵۵ شناسه × چهار خانواده** دارد؛ `REPORTABLE_NOW_AS_HISTORICAL=90`، `QA76_GATED=44`، `P2_INFERENCE_REQUIRED=36`، `PERMANENT_NOT_APPLICABLE_WITHOUT_NEW_QUERY_START_CONTRACT=42`، `OPTIONAL=8`. این طبقه‌بندی **وضعیت زمان تهیه P3** را محفوظ می‌دارد، نه تصمیم جدید اداری.
- **رویداد علمی بعدی (بدون ویرایش P1/P3):** رأی ممیز `ACCEPT_QA76_CENTER_ONLY` در https://github.com/mahhammad/My-Thesis/issues/49#issuecomment-6796097800 فقط **۷۶ مرکز و ۸۳۶ مقدار center-only** را تأیید کرده است؛ ۴۴ شناسهٔ دیگر خودبه‌خود verified-v2 نشده‌اند. در شاخه دائمی، برچسب P3 `QA76_GATED` و عبارت `PENDING_INDEPENDENT_QA76` در P1 عمداً **به‌عنوان provenance نسخه اولیه بدون overwrite** باقی مانده‌اند. برای تفسیر جاری، رأی ثبت‌شده در Issue مرجع است.
- `R_KL^{legacy,clipped}` با exact-support-aware KL برابر اعلام نشده؛ `R_sqrt_JS` میانگین sqrt(JS) نمونه‌ای است، نه sqrt(mean JS)؛ near-tie با exact-tie یکی نیست. Validation در ۲۶ اجرای `node/node_class` بدون query-start برابر **NOT_APPLICABLE** است، نه صفر. `P2_MINIMAL_DECISION.md` فقط طرح، **P2 = NOT_AUTHORIZED**.
- گزارش‌های قدیمی اختلاف زمان اجرا، `discrepancy_report.md` و boundary fixtureها بدون تصحیح، حذف یا ترفیع اعتبار حفظ شده‌اند. شواهد منفی و `not_measured` بدون پرکردن مصنوعی محفوظ هستند.

## ۴) Research Journal / Issue #54

**۲۸ فایل Git blob در `research_journal/` در هر سه tree یکسان‌اند**؛ هیچ افزودن/تغییر/حذف ژورنال در این زنجیره رخ نداده است. فهرست [Issue #54](https://github.com/mahhammad/My-Thesis/issues/54) همچنان مرجع ثبت موضوعی یادداشت‌هاست؛ ژورنال جدیدِ منحصربه‌فردی در این سه شاخه برای انتقال شناسایی نشد. فایل‌های `p1_provisional/` و `p3_publication_gates/` گزارش شواهد آزمایش‌اند، نه ژورنال جدید و نه ادعای مستند شدن علمی در شاخه chapter4.

## ۵) نگه‌داری شاخه‌ها، PR و گیت حذف

- **سه source branch:** هر سه `CANDIDATE_FOR_DELETION_AFTER_ARCHIVE_AND_QA` و هم‌زمان **`KEEP_UNTIL_ARCHIVE_VERIFIED`**؛ *در این مأموریت هیچ‌کدام حذف، force-push یا rename نشده‌اند*. از نظر محتوایی هیچ فایل اختصاصی خارج از زنجیره P3 شناسایی نشد (`ARCHIVED_ONLY=0`، `NOT_COVERED=0` برای این سه درخت)، ولی مجوز حذف تنها بعد از حفاظت واقعی تاریخچه و QA مستقل است.
- هنگام کنترل این مأموریت، شاخه `archive/thesis-full-20261112` در GitHub **پیدا نشد**: `ARCHIVE_VERIFIED=NOT_YET_EVIDENCED`. QA نهایی باید archive HEAD مستقل را از نظر ancestor بودن هر سه source SHA و بازیابی بایت/blob شواهد آزمایش کنترل کند؛ فقط وجود این Markdown کافی نیست. تا گیت `ARCHIVE_VERIFIED`، `INDEPENDENT_TREE_QA=PASS` و مجوز صریح مالک، **`DELETE=NO`**.
- **PR #51** با مبدأ `editorial/ch4-sec412-benchmark-r03` و مقصد `chapter4`، در شروع `open/draft` و `merged=false` بود؛ محتوای آن به شاخه دائمی منتقل نشد، چون نگارش توسط ویراستار در شاخه جدا ادامه دارد. PR #51، `chapter4` و سایر branchها **دست‌نخورده‌اند**؛ بدون ایجاد PR یا merge جدید.
- آزمایش/solver/optimizer/train/backfill/replay/Test/CI انجام نشده، metric یا مرکز محاسبه نشده، ترفیع اعتبار علمی صورت نگرفته است.

**مسیر QA مستقل پیشنهادی:** `git cat-file -e <SHA>^{commit}` برای هر source، `git merge-base --is-ancestor <SHA> researcher/03-chapter4` برای هر سه، `git rev-parse <ref>^{tree}` و `git ls-tree -r` برای snapshotها، یکسان‌بودن SHA blobهای raw/CSV/JSON/report و ۹ فایل افزوده، شمارش مستقل ۷۶/۵۵/۲۲۰ و تأیید جداگانهٔ archive HEAD. هر mismatch => `STOP/KEEP_BRANCH`.

**رأی:** `READY_FOR_INDEPENDENT_TREE_QA / OLD_BRANCHES_NOT_DELETED`. — **RESEARCHER-03**
