# گزارش فنی و پروندهٔ شواهد — E-CH4-12 | مقایسهٔ استاندارد خانواده‌های سیاست (۷۶ اجرا)

**منبع مبنا:** commit `5ddce088204b` | **قرارداد:** فصل چهارم پایان‌نامه و پرونده‌های ماشین‌خوان داخل همین مطالعه

## چکیده، دامنه و تعریف شواهد

این مطالعه شامل ۷۶ اجرای ثبت‌شده از چهار خانوادهٔ سیاست و شواهد مرکز-محور پذیرفته‌شده است. ۷۶ مرکز و ۸۳۶ مقدار سنجهٔ مرکز independently accepted هستند؛ این رأی به سایر معیارهای گزارش تعمیم داده نمی‌شود. گزارش R03 و شواهد نگارشی با منشأ تاریخی در همین سند ادغام می‌شوند.

## شرح علمی و قراردادهای ارزیابی

## پیوست الکترونیکی §۴.۱۲ — مقایسه استاندارد خانواده‌های سیاست

**شناسه مطالعه:** \`E-CH4-12\` | **وضعیت:** \`EXECUTED / EVIDENCE_ARCHIVED\`  
**متن اصلی:** فصل چهارم، §۴.۱۲، برچسب \`sec:ch4-benchmark-v1\`.  
**کارکرد این پوشه:** بستهٔ مستقل شواهد همراه پایان‌نامه، بدون وابستگی لازم به نام یا مسیر پوشهٔ پژوهشی اولیه. فایل‌های خام و خلاصه‌ها بدون دستکاری محتوا از آرشیو مطالعه به این پوشه **کپی** شده‌اند؛ نسخه توسعه‌ای برای کار پژوهشی محفوظ است.

### بازآرایی شواهد R03

ابزار بازسازی و شواهد علمی مختص این مطالعه اکنون در [پوشهٔ R03](README_FA.md) قرار دارند؛ [مدرک نگارش و حدود رأی ممیزی](README_FA.md) نیز حفظ شده است. خروجی‌های میانی و اسکریپت‌های یک‌بارمصرف از نسخهٔ فعال کنار گذاشته شده‌اند و در تگ `backup/pre-cleanup-2026-11-15` قابل بازیابی هستند. داده‌های رسمی پوشهٔ `results/` تغییر نکرده‌اند.

### مطالعه و شواهد

۷۶ اجرای ثبت‌شده به چهار بلوک تقسیم می‌شوند: ۱۶ اجرای اصلی ۵۰گامی، ۱۲ اجرای هدف جایگزین ۲۵گامی، ۲۴ اجرای اندازه مرجع ۲۵گامی در مولد پایه و ۲۴ اجرای دیگر ۲۵گامی با دو مولد مستقل. این ۷۶ اجرا **۷۶ تکرار مستقل مولد نیستند**. چهار خانواده \`shared\`، \`class\`، \`node\` و \`node_class\` ارزیابی شدند؛ در قرارداد موجود، Validation برای دو خانواده گره‌ای \`not_applicable\` است، نه صفر.

از ۷۶ مرکز ذخیره‌شده، **۱۱ شناسه سنجه مرکز × ۷۶ اجرا = ۸۳۶ مقدار مستقل از بازنمایی نمونه‌ای ممیزی و پذیرفته شده‌اند**. پذیرش این سنجه‌ها را به BA، NLL یا KL نمونه‌ای تعمیم ندهید؛ پوشش هر شناسه به صورت جداگانه در مانیفست ثبت شده است.

### راهنمای پرونده

| پرونده | استفاده |
|---|---|
| [PROTOCOL_FA.md](README_FA.md) | مدل، مولد، بودجه و محدودیت‌های استنتاج |
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

#### آرشیو خام خروجی‌ها

| فایل در \`results/\` | تعداد اعضای اجرا | SHA-256 آرشیو |
|---|---:|---|
| \`core_50_16runs.zip\` | ۱۶ | \`9609eb96c207316fde26934cbef4668ba287a275d4485175d1da7e798a297040\` |
| \`extra_25_12runs.zip\` | ۱۲ | \`061a00923ae69ac2f8b923947bd9cdea9d4f3dba1d1771063bcb3ecffc0201e3\` |
| \`size_25_24runs_seed20261102.zip\` | ۲۴ | \`d570fb3f1dcab66476f027e1ade17b4e87c7ca16ff7fa6c76fda87b260c84e9f\` |
| \`size_25_24runs_seeds20261103_20261104.zip\` | ۲۴ | \`fdb862462e237e5094204ff17a5bc1a79a0f1d17f3a9b9d90ce967dba8b11fdd\` |

هر آرشیو حاوی فایل‌های JSON اجراها، پارامترهای سیاست، مراکز و خروجی‌های ذخیره‌شده است. در جدول \`master_runs_76.csv\` ستون \`file\` نام عضو JSON و ستون \`sha256\` هش همان عضو است. چهار ZIP برای بررسی شواهد اصلی کافی‌اند، ولی **ادعای اجرای مستقل تمام آموزش‌ها از ابتدا** از وجود این فایل‌ها حاصل نمی‌شود.

**قرارداد مهم:** \`R_KL\` تاریخی دارای clipping محاسباتی است و همان KL دقیقِ \`support-aware\` نیست. \`R_sqrt_JS\` میانگین جذر JSD نمونه‌هاست، نه جذر میانگین JSD. داده آزمون برای انتخاب \`τ\` و سیاست دوباره خوانده نشده است.

این بسته برای توزیع به‌عنوان پیوست الکترونیکی قابل انتخاب است. فایل‌های پژوهشی و کدهای توسعه لازم نیست همراه این بسته به خواننده داده شوند، مگر آنکه بازاجرای کامل آموزش جداگانه خواسته شود.


## پیوست پژوهشی: PROTOCOL_FA.md

> این بخش بخشی از منشأ پژوهش است. تاریخ‌ها، بودجه‌ها، شواهد مرحله‌ای و محدودیت‌ها باید با نسخهٔ فعلی و رأی علمی مستقل مقایسه شوند.

## قرارداد مطالعه §۴.۱۲: پروتکل، داده‌ها و حدود استنتاج

**مأموریت:** \`E-CH4-12\` — نسخهٔ شواهد معتبر، بدون بازآموزی جدید.

### پرسش

آیا در قرارداد ثابت، بیشینه‌سازی جدایی مراکز یا کاهش پراکندگی درون‌کلاسی به‌تنهایی موجب بهبود Balanced Accuracy می‌شود؟ میزان انعطاف سیاست و تفاوت بودجه‌های آموزشی در تفسیر لحاظ می‌شوند. این مطالعه ادعای بهینه‌بودن سراسری هیچ خانواده سیاست را ندارد.

### ورودی و سیاست‌ها

داده پایه: سه کلاس، پنج ویژگی؛ Train 300، Validation 90؛ اندازه مرجع در بررسی اندازه‌ها \`{50,100,150,200,250,300}\`، سه بذر مولد \`20261102,20261103,20261104\`. زیرمجموعه‌ها در یک مولد تو‌در‌تو هستند. خانواده‌ها: \`shared (Z)\`، \`class (Z_c)\`، \`node (Z_i)\` و \`node_class (Z_{c,i})\`. انتقال عمودیِ بدون بازگشت به مبدأ و نرمال‌سازی روی مقصد مجاز؛ \`β=0.4, λ=0.8, H=12\`، امید دقیق و \`float64\`.

در اجرای اصلی: مقداردهی اولیه صفر، \`Adam\`، نرخ یادگیری \`0.12\`. ۱۶ اجرا با \`n=300\` و ۵۰ گام از هدف جدایی \`I\` و سه پهنای margin-JSD \`τ={0.002,0.005,0.020}\`. ۱۲ اجرای هدف‌های sqrtJSD/KL/ترکیبی با ۲۵ گام؛ ۴۸ اجرای اندازهٔ مرجع با ۲۵ گام (۲۴ اجرای چهار خانواده با یک seed، ۲۴ اجرای shared/class با دو seed دیگر). ردیف‌های دو بودجه را هم‌پروتکل ۵۰گامی تصور نکنید.

### بازنمایی، Validation و سنجه‌ها

مرکز هر کلاس از بازنمایی Train همان سیاست و همان مرجع محاسبه و برای Validation ثابت نگه داشته می‌شود. \`shared/class\` دارای شروع برون‌گرافی‌اند؛ \`node/node_class\` در قرارداد فعلی شروع برون‌گرافی ندارند؛ Validation آن‌ها **N/A** است. گزارش امتیاز آموزش ۱۰۰٪ یک خانواده گره‌ای، نتیجه تعمیم آن نیست.

سنجه‌ها: Balanced Accuracy و در صورت دسترس Accuracy/Confusion، \`R_JS\`, \`R_sqrt_JS\` و KL تاریخی، جدایی \`T, BT, TNL, BTNL, I\`، کمینه/میانگین فاصلهٔ √JSD مراکز، ماتریس‌های مرکز و سنجه‌های دیگر با وضعیت مشخص در \`coverage_manifest_76.json\`. ممیزی مستقل ۸۳۶ مقدار فقط \`center-only\` را تأیید کرده؛ BA و R نمونه‌ای در بسیاری موارد گزارش‌های ذخیره‌شده با کنترل تجمیع هستند. استفاده از Test برای مدل‌گزینی جدید ممنوع است.

### بازخوانی نتایج

۱. ابتدا \`master_runs_76.csv\` را بر اساس \`family,objective,tau,n,steps,data_seed\` فیلتر کنید؛ \`file\` و \`sha256\` عضو JSON را مشخص می‌کنند.
۲. عضو JSON را از یکی از چهار آرشیو \`results/*.zip\` استخراج و هش SHA-256 را کنترل کنید.
۳. پیکربندی، تاریخچه سیاست، مراکز و سنجه‌های ذخیره‌شده را از JSON بخوانید.
۴. برای مقایسه هندسهٔ مراکز، \`center_only_metrics.csv\` و \`coverage_manifest_76.json\` را ملاک قرار دهید؛ وجود مقدار در CSV به معنی ممیزی تمام شناسه‌ها نیست.
۵. جداول §۴.۱۲ خلاصه مقایسه علمی‌اند، و بسته حاضر جزئیات شواهد را ارائه می‌کند. بازاجرای کامل optimizer یا پیاده‌سازی موتور گشت به بسته توسعه‌ای جدا نیاز دارد.


## پیوست پژوهشی: r03_center_recovery/README_FA.md

> این بخش بخشی از منشأ پژوهش است. تاریخ‌ها، بودجه‌ها، شواهد مرحله‌ای و محدودیت‌ها باید با نسخهٔ فعلی و رأی علمی مستقل مقایسه شوند.

## بازیابی هندسهٔ مراکز — پژوهشگر R03 / E-CH4-12

این زیرپوشه **ویژهٔ مطالعهٔ §۴.۱۲ (مقایسهٔ استاندارد خانواده‌های سیاست)** است و ابزار عمومی تمام آزمایش‌ها محسوب نمی‌شود.

### منابع معتبر

- شواهد رسمی: [۷۶ ردیف هندسهٔ مراکز](results/center_only_metrics.csv)، [مانیفست ۷۶ اجرا](results/coverage_manifest_76.json) و [ممیزی بازیابی](results/recovery_audit.json).
- رأی مستقل [`ACCEPT_QA76_CENTER_ONLY`](https://github.com/mahhammad/My-Thesis/issues/49#issuecomment-6796097800) فقط **۷۶ مرکز و ۸۳۶ مقدار مرکز-محور** را پوشش می‌دهد؛ این حکم به ۴۴ شناسهٔ دیگر معیار تعمیم نمی‌یابد.
- [مدرک نگارش §۴.۱۲](README_FA.md) و [گزارش محدودیت‌ها و اختلاف‌های R03](README_FA.md) با Git blob اصلی و بدون تغییر بایت منتقل شده‌اند.

### ابزارهای بازتولید

- `center_only_rebuild.py`: بازسازی فقط معیار هندسهٔ مراکز از چهار ZIP رسمی، با کنترل SHA-256 آرشیو و اعضا؛ بدون آموزش، بازپخش گشت و دسترسی Test؛ **پیش‌فرض dry-run**.
- `test_backfill_boundaries.py`: آزمون مرزهای KL، softmax ماسک‌شده، tie و هندسهٔ مرکز.
- `boundary_fixture_results.json`: شاهد تاریخی fixture مستقل؛ به‌معنای موفقیت CI یا ممیزی کامل نیست.

از ریشهٔ مخزن:
```bash
python code/experiments/standard_policy_comparison/r03_center_recovery/center_only_rebuild.py
python code/experiments/standard_policy_comparison/r03_center_recovery/center_only_rebuild.py --execute --out-dir /tmp/ch4-r03-centers
python -m unittest discover -s code/experiments/standard_policy_comparison/r03_center_recovery -p 'test_backfill_boundaries.py' -v
```

### پرونده‌های تاریخی بازنشسته از checkout

جدول و مانیفست اولیهٔ **۳۶/۷۶** شامل `center_only_metrics.csv`، `coverage_manifest.csv/json`، `coverage_summary.csv` و `verified_sources_36.json` دیگر منبع جاری نیستند. هر ۳۶ اجرا و تمام ۲۵ ستون جدول اولیه با جدول رسمی ۷۶تایی مطابقت دارند (بیشینهٔ اختلاف عددی حدود `8.61e-16`، بدون مغایرت فراتر از `1e-10`). این تطبیق **فقط داده‌های هندسهٔ مراکز** را پوشش می‌دهد.

اسکریپت‌های مختص اجرای یک‌بارهٔ ترمیم `run_recovery.py`، `preflight_recovery.py`، `verify_recovery.py`، `budget_guard.py`، ابزار `zip_extract.js` و گزارش اداری `CONSOLIDATION_R03_FA.md` نیز از نسخهٔ فعال کنار رفته‌اند.

**بازیابی نسخهٔ اولیه:** [تگ قفل‌شدهٔ پیش از پاک‌سازی](https://github.com/mahhammad/My-Thesis/tree/backup/pre-cleanup-2026-11-15/code/experiments/benchmark_v1/report_backfill_v2). فایل‌های رسمی `results/`، متن فصل‌ها و نتایج علمی تغییر نکرده‌اند. مسیرهای قدیمی در اسناد پژوهشی تاریخی، ارجاع منشأ بوده‌اند و نباید به‌عنوان مسیر اجرایی فعال تفسیر شوند.


## پیوست پژوهشی: r03_center_recovery/discrepancy_report.md

> این بخش بخشی از منشأ پژوهش است. تاریخ‌ها، بودجه‌ها، شواهد مرحله‌ای و محدودیت‌ها باید با نسخهٔ فعلی و رأی علمی مستقل مقایسه شوند.

## [RESEARCHER-03][BACKFILL] CH4-METRICS-BACKFILL-01 — discrepancy and coverage report

**Author:** Researcher No. 3 / پژوهشگر شماره ۳ · **date:** 2026-11-11  
**Immutable reference:** `chapter4@de02fdda91e23db177de4429b3e1a9693c046d83`  
**Delivery branch:** `experiment/ch4-metrics-backfill-v2` · **contract:** `chapter4-common-evaluation-contract-v2`  
**Scope:** 76 historical Benchmark v1 checkpoint records. Zero optimizer steps; zero walk inference replays; zero Test access. Original ZIPs and `chapter4/main/tex` unchanged. This report is **not independent QA**.

### A. Manifest / provenance results

- `coverage_manifest.json`: exactly **76 unique run_id** with 55 metric status/reason entries each, versions and source-member SHA. `coverage_manifest.csv`: exactly **76 data rows** and 129 columns; `coverage_summary.csv` enumerates status counts per metric. Status vocabulary: `measured/not_measured/not_applicable/undefined_or_infinite/not_verified`. Null/NA is never replaced by zero.
- Parsed `master_runs_76.csv` and `posthoc_score_backfill_76_compact.csv`, each containing 76 source rows. Verified family distribution **shared 25, class 25, node 13, node_class 13** and **50 numeric Validation BA / 26 undefined query-start**.
- Direct original ZIP retrieval and verification covered **36 / 76** records:
  - `extra_25_12runs.zip` (**12**), bytewise SHA-256 `061a00923ae69ac2f8b923947bd9cdea9d4f3dba1d1771063bcb3ecffc0201e3`.
  - `size_25_24runs_seeds20261103_20261104.zip` (**24**), SHA-256 `fdb862462e237e5094204ff17a5bc1a79a0f1d17f3a9b9d90ce967dba8b11fdd`.
  - For **all 36**, computed each raw JSON SHA-256 and confirmed equality to **both** its archive manifest entry and the master CSV member digest; matched family, objective, steps, data seed, `test_used=false`, valid 3×3 centers.
  - `core_50_16runs.zip` (**16**) and `size_25_24runs_seed20261102.zip` (**24**) are present in the GitHub tree / historical source registry, but the current GitHub binary retrieval returned **empty data**. The separate Library copy could not be materialized. **These 40 were NOT reverified from raw bytes**; their historical scalar metrics remain version-labelled CSV evidence, not regenerated center metrics. Do not infer missing centers from the summary.
- `verified_sources_36.json`: exact 36 verified center matrices, original checkpoint summaries, member digests and provenance. `center_only_metrics.csv`: **36 numerical center-only rows** with `T,BT,TNL,BTNL,I`, all pairwise JS/sqrtJS and similarity matrices, per-class nearest sqrtJS, min/mean pair distances. **40 rows are deliberately absent**, not zero.

### B. Numeric checks and discrepancies

**No positive discrepancy detected among the 36 accessible center records.** Recomputed center `I`, `min_pairwise_sqrt_JS`, and `mean_pairwise_JS` agree with the original final Train summary to observed floating-point exact agreement in the JavaScript computation (`max |Δ|=0` in these three comparisons). Independently comparing original raw values to master CSV yielded maximal discrepancies `|ΔI|=9.03e-17`, `|ΔR_JS|=9.72e-17`, `|ΔBA|=1.12e-16`, fully attributable to roundoff. **Not checked:** T/BT/TNL/BTNL for the unavailable 40, all unscanned per-sample arrays, and full 76-run v2 congruence. The `center_only_rebuild.py` script calls the **canonical** Chapter-3 NumPy `center_metrics` from `code/policy_evaluator/metrics.py` after checking both ZIP and member SHA. It is **dry-run by default**, reads no Test, and performs no walk inference/training; it can regenerate all 76 center-only values when all four archives are locally accessible.

**KL support certificate on 36, strictly limited scope:** the minimum entry across **every class center in these 36 raw verified checkpoints** is **0.024847449072930662**, greater than zero and far larger than the legacy clipping floor `1e-300`. Thus no valid candidate representation `P` can have `P_j>0,V_{c,j}=0` against *these saved center matrices*; the particular `+∞` versus clipped-KL **support mismatch cannot arise for these 36 centers** (Train or defined Validation). This is a **center-only mathematical certificate**, not an empirical enumeration of all candidate `P`, nor a precise re-evaluation of the KL sums. The historical numeric `R_KL` is retained, tagged `legacy_clipped_numerically_support_compatible_not_replayed`; exact-support KL candidate scores and NLL are not newly measured. The **remaining 40** lack a verified center-support certificate and are `not_verified` under primary v2 KL.

**Tiny independently executed expression-equivalent fixture:** using NumPy 2.3.5 / Torch 2.10.0+cpu, `KL((1,0)||(0,1))` is `+∞` in exact contract while legacy floor produces **690.7755278982137**; all-invalid masked softmax produces `[.5,.5]`; explicit valid mask produces `[1,0]`; positive-mass invalid row would leak choice while zero-mass row contributes zero. Tie/near-tie fixtures verify that `argmax`, `isclose(atol=1e-12)` and `margin<=0` are **not equivalent**, as documented in `boundary_fixture_results.json`. The fixture completed in **0.00534 s**, peak whole-process RSS **253.31 MiB**; it **reimplemented the checked expressions**, it was *not* execution of the imported repo unit test. A real import-based regression suite is provided as `test_backfill_boundaries.py` but has **NOT** been run in CI in this branch. No engine patch was applied.

### C. Important metric coverage counts

| Metric group | measured | not_verified | not_measured | not_applicable |
|---|---:|---:|---:|---:|
| each center-only metric `T/BT/TNL/BTNL/I/pairwise...` | 36 | 40 | 0 | 0 |
| Train `R_JS`, Train `R_sqrt_JS` (legacy-v1 numbers) | 76 | 0 | 0 | 0 |
| Train **primary** `R_KL` (verified center support) | 36 | 40 | 0 | 0 |
| Train `R_KL_legacy_clipped` (reported) | 76 | 0 | 0 | 0 |
| Train BA | 76 | 0 | 0 | 0 |
| Train ordinary accuracy / confusion / margin quantiles / sigmoid | 36 | 40 | 0 | 0 |
| Train JS-NLL / posthoc Softplus(τ=.005) / historical KL-NLL | 76 | 0 | 0 | 0 |
| Train exact/near `tie_count` | 0 | 0 | 76 | 0 |
| Validation BA | 50 | 0 | 0 | 26 |
| Validation `R_JS/R_sqrt_JS/R_KL` | 30 | 20 | 0 | 26 |
| Validation JS-NLL / posthoc historical KL-NLL | 50 | 0 | 0 | 26 |
| Validation ordinary accuracy/margins | 30 | 20 | 0 | 26 |
| Validation exact/near `tie_count` | 0 | 0 | 50 | 26 |

`measured` encompasses a **number actually recorded in the historical CSV**; it does **not imply we independently retrieved/checksummed that original member**. Per-run `source_evidence` disambiguates the 36 SHA-verified raw records from 40 old CSV-only records. Historical scores and `NLL_KL_normalized_historical` are **not** the `η,ε`-smoothed stable KL score from Chapter 3. All Softplus values imported from `posthoc_score_backfill_76_compact.csv` use **posthoc τ=.005**, which must not be confused with a checkpoint trained with τ=.002 or .020.

**Other prohibitions:** 50-step and 25-step budgets are separately labelled, size-study subsamples of one data seed are correlated rather than independent repetitions; Oracle Bayes posterior geometry is distinct from walk-neighborhood geometry; node/node_class Validation has no defined unseen-query start and must remain `not_applicable`; historical Test cannot select models.

### D. BLOCKER / EDITORIAL / FUTURE / next gate

- **BLOCKER for claiming 76 fully regenerated v2 center reports:** 40 archival members were not actually retrieved in this session. Remedy: independent QA executes **only** `center_only_rebuild.py --execute` from the source-locked checkout with all four local ZIPs, verifies 76/76 source SHA + reference formula matches, and adds the derived 40 without overwriting historical sources. This is **center-only** and needs **no inference**. The current 36-row CSV must **not** be presented as a 76-row measured center set.
- **EDITORIAL:** maintain historical-v1 versus center-v2 versions, legacy clipped versus support-aware KL, `near_tie` versus exact tie, `hard-margin error` versus `1-BA`, posthoc τ and objective τ, and explicit null/not-applicable states in §4.12. No LaTeX modified.
- **FUTURE, not authorized by this report:** after independent QA decide whether to permit bounded inference-only replay to measure per-sample exact KL/near ties; the current mission used **0 of 2** allowable inference replays and **0 of 180 seconds** replay budget. No new training, Test usage, or algorithm ranking.
- **Mask proposal only:** on future authorized patch, make all-invalid masked policy rows explicitly zero mass and fail fast if reachable with positive state mass; keep current graph validity and normalization unchanged until compatibility tests pass.

**Recommended handoff:** `PARTIAL_WITH_REASONS`, with 76/76 manifest records and 36/76 SHA-verified center-only updates. Independent QA should first secure byte access to the remaining two historical ZIPs and execute checksum-guarded **center-only** backfill; no automatic broad inference replay is scientifically justified.

**RESEARCHER-03 — پژوهشگر شماره ۳.**


## پیوست پژوهشی: r03_center_recovery/section412_editorial_evidence/README_FA.md

> این بخش بخشی از منشأ پژوهش است. تاریخ‌ها، بودجه‌ها، شواهد مرحله‌ای و محدودیت‌ها باید با نسخهٔ فعلی و رأی علمی مستقل مقایسه شوند.

## شواهد نگارشی §۴.۱۲ — Benchmark v1

> **مسیرهای معتبر پس از سبک‌سازی (۱۵ نوامبر ۲۰۲۶):** چهار ZIP و جدول‌های اصلی ۷۶ اجرای §۴.۱۲ در `code/experiments/standard_policy_comparison/results/` محفوظ‌اند. نتایج منحصربه‌فرد تاریخی و گزارش‌های مرحله‌ای R03 با Git blob اصلی در تگ قفل‌شده `backup/pre-cleanup-2026-11-15` قابل بازیابی‌اند؛ نسخه فعال دیگر `code/experiments/archive/` ندارد. اسکریپت‌های بازسازی از مسیر رسمی ZIP استفاده می‌کنند؛ گزارش‌های قدیمی زیر، وضعیت تاریخی زمان نگارش خود را ثبت می‌کنند و لزوماً مسیر فعلی فایل‌ها را نشان نمی‌دهند. راهنمای مسیرهای بازیابی در `research_journal/archive_recovery/README_FA.md` آمده است.


**دامنه:** فقط مرجع عددهای §۴.۱۲ در `tex/chapter4.tex` در مأموریت `CH4-R03-SEC412-EDITORIAL-PR-01`. این فایل، متن پایان‌نامه نیست و برای ثبت provenance، نسخهٔ معیارها و تصمیم‌های علمی کنار آرشیو نگهداری می‌شود. هیچ آرشیو یا خروجی آزمایش بازنویسی نشده است.

### منابع ثابت

- شاخه مرجع فصل چهارم: `chapter4@de02fdda91e23db177de4429b3e1a9693c046d83`؛ فایل `tex/chapter4.tex` پیش از ویرایش Git blob `b341e38fcc44b9487a9e2f0ad7c2ac0a9a370637`.
- فهرست ۷۶ اجرا: `code/experiments/benchmark_v1/results/master_runs_76.csv`، Git blob `bcc2d7f4a4f39301973eb53c465ff05a6b4d0f21`. چهار ZIP آرشیو در `results/` ثابت هستند.
- ۱۱ معیار هندسهٔ مراکز: `code/experiments/benchmark_v1/report_backfill_v2/full_76_candidate/center_only_metrics.csv` در `experiment/ch4-metrics-backfill-v2@0a552e6e31999f0a9aeab499f62d3b54313c2e1c`، Git blob `4e7ab942ea6a825639829b868e41ed50d654d14d`.
- [رأی علمی مستقل `ACCEPT_QA76_CENTER_ONLY`](https://github.com/mahhammad/My-Thesis/issues/49#issuecomment-6796097800): ۷۶/۷۶ مراکز و ۸۳۶/۸۳۶ مقدار center-only پذیرفته شده‌اند؛ نه ۴۴ شناسهٔ دیگر.
- اطلاعات اعتبارسنجی aggregate تاریخی و منشأ: `experiment/ch4-metrics-p1-provisional-r03@d36ec6dc522e4e6d5f7ee4134f23c71f97b2aff3`، Git blob جدول `raw_aggregate_audit_76.json@d6d83e9937b58b68aa40cecc3994bf2542595a79`؛ `coverage_76x55_provisional.csv@41e2d2f3252fd5b3145c9215f6aee8d225ab9dea`.
- `code/experiments/COMMON_METRICS_CONTRACT.json` در base، blob `a7e1cea6276a339053a8ded2dfb56d2ad334f903`. فصل سوم منبع تعریف $T,BT,TNL,BTNL,I$، JS و زیان حاشیه است.
- [دستور محدودکنندهٔ ویرایش و Draft PR](https://github.com/mahhammad/My-Thesis/issues/49#issuecomment-6796729350).

### محل دقیق هر نتیجه

| جدول §۴.۱۲ | دامنه | منشأ | سطح شواهد |
|---|---|---|---|
| `tab:ch4-v1-I-margin-main` | هشت اجرای $n=300$, 50 steps (max-I و margin-JSD با tau=.005) | `master_runs_76.csv` و JSON raw اعضای ثبت‌شده با SHA زیر | Train/Validation BA و R نمونه‌محور: **historical source + internal aggregate consistency**، نه independently verified v2 |
| `tab:ch4-v1-center-geometry` | همان هشت اجرا؛ $T,BT,TNL,BTNL,I$، min/mean sqrtJS و mean JS میان مراکز | `center_only_metrics.csv` ثابت و رأی `ACCEPT_QA76_CENTER_ONLY` | **independently accepted center-only** |
| `tab:ch4-v1-tau` | دوازده اجرای حاشیه JSD با n=300، 50 steps، tau در {.002,.005,.020} | `master_runs_76.csv`، `final.train` و Validation برای دو خانوادهٔ مجاز | **توصیفی با BA جمع‌بندی‌شده**؛ node-family Validation ندارد |
| `tab:ch4-v1-size-three-seeds` | 36 اجرای `shared/class` در سه data-seed و شش n، 25 steps | `master_runs_76.csv`؛ میانگین و انحراف معیار نمونه‌ای از سه مقدار Validation BA موجود برای هر (family,n) | **آمار توصیفیِ سه مولد مستقل**؛ بدون آزمون فرضیه یا برآورد uncertainty دقیق |
| متن هدف‌های جایگزین | ۱۲ اجرا با n=300، 25 steps، اهداف sqrtJSD, KL, ترکیبی | `master_runs_76.csv`؛ 4 خانواده × 3 هدف | BA و R همان نسخه ذخیره‌شده؛ **بدون قیاس بودجهٔ 25/50** |

**اعضای هشت اجرای جدول هندسه/عملکرد** (برای کنترل پیوند ۱ به ۱ run و SHA):

| JSON عضو آرشیو | SHA-256 member | خانواده | هدف |
|---|---|---|---|
| `ds20261102_n300_class_margin_JSD_tau0.005_s50.json` | `a407166cb68906c97769de2fc6618dfd058d05cd1eb189e794e12c69eeea87db` | `class` | `margin_JSD` |
| `ds20261102_n300_node_margin_JSD_tau0.005_s50.json` | `51243e1fe1da0e97265d4246df6831eda2274b2b48509b6774f513bc7afe7a94` | `node` | `margin_JSD` |
| `ds20261102_n300_node_class_margin_JSD_tau0.005_s50.json` | `f3d977e94af217b3a20f8c07134286cf3dbee4b7ffb7bbc2520e118163dfdf72` | `node_class` | `margin_JSD` |
| `ds20261102_n300_shared_margin_JSD_tau0.005_s50.json` | `8fa82f9cb613df31a837c6a26f771f81a45f9dc3dada1f68d479b66c70f4c34c` | `shared` | `margin_JSD` |
| `ds20261102_n300_class_max_I_tau0.005_s50.json` | `ed09ba752b1ecc87354db418722a6e8cb385b1d743ec30bd79434f623753298c` | `class` | `max_I` |
| `ds20261102_n300_node_max_I_tau0.005_s50.json` | `6041a9862f2d2450b0b6f5607c750d4b659decd630a6240c55a221456950c725` | `node` | `max_I` |
| `ds20261102_n300_node_class_max_I_tau0.005_s50.json` | `5f3b99619a5d69414137768e34ba733e0fdf08a10d024993a2c282effda3b904` | `node_class` | `max_I` |
| `ds20261102_n300_shared_max_I_tau0.005_s50.json` | `e8018e579b20594869c15a9be9f27f681afed2cd461edf510468bfa1e4e3184b` | `shared` | `max_I` |

#### قرارداد نسخه معیارها

- `center.T/BT/TNL/BTNL/I/pairwise_*` بر مبنای مراکز Train هستند؛ تأیید مستقل تنها همین ۱۱ شناسه را دربرمی‌گیرد.
- `train.R_JS` و `train.R_sqrt_JS` (و Validation دارای آغاز مجاز) از دادهٔ اصلی ثبت شده و **حساب aggregate کلاسی‌شان کنترل شده است**، اما محاسبهٔ JSD تک‌نمونه‌ای از checkpoint سیاست در این مأموریت انجام نشد. `R_sqrt_JS` یعنی میانگین ریشهٔ JSD در سطح نمونه، نه `sqrt(R_JS)`.
- ستون `R_{KL}^{legacy,clip}` در §۴.۱۲ عین `R_KL` ذخیره‌شدهٔ Benchmark اولیه است که در `code/experiments/benchmark_v1/metrics.py` clipping عددی دارد؛ **آن را به `R_KL_exact_support_aware` قرارداد v2 تبدیل نکنید**. اعداد صدکی/حاشیه/NLL و ties فقط با provenance و در جای مناسب خود گزارش شوند؛ QA76 آن‌ها را تأیید نکرده است.
- هر `node/node_class` برای نمونهٔ خارج از گراف سیاست شروع قراردادشده ندارد: ۲۶ اجرای مربوطه و همهٔ سنجه‌های Validation آن‌ها **not_applicable** هستند. اعداد Train به معنی قابلیت تعمیم نیستند.
- `experiment_specific.elapsed_s` در raw، زمان پایان آماده‌سازی خروجی `run.elapsed_s` است؛ `final.walltime_s` زمان آخرین history با گردکردن چهار رقم اعشار. اختلاف timestamp در ledger P1 باقی می‌ماند.
- `posthoc_score_backfill_76_compact.csv` اعداد NLL/Softplus شش‌رقمی دارد؛ نه calibrated Bayes posterior و نه independently verified-v2. متن آنها در بخش بعدی فصل بررسی می‌شود، نه در §۴.۱۲.

#### انتقال جدول آزمون پیشین از متن اصلی به سند شواهد

این جدول تنها برای **حفظ منشأ نسخهٔ قبلی §۴.۱۲** کپی شده و **ورودی انتخاب سیاست، انتخاب tau، مقایسهٔ علمی جدید یا آزمون مستقل نیست**. مرجع قبلی `tex/chapter4.tex@de02fdda91e23db177de4429b3e1a9693c046d83`، جدول `tab:ch4-v1-historical-heldout`. هیچ دادهٔ Test دوباره خوانده یا اجرا نشده است.

| سیاست | Train BA % | Validation BA % | Test BA پیش‌تر گزارش‌شده % |
|---|---:|---:|---:|
| یکنواخت | 83.65 | 96.77 | 76.42 |
| حاشیه tau=.002 | 87.21 | 96.60 | 78.37 |
| حاشیه tau=.005 | 86.52 | 97.28 | 78.37 |
| حاشیه tau=.020 | 81.02 | 90.14 | 82.39 |
| پسین بیزی واقعی مولد | 84.32 | 97.45 | 83.88 |

**علت حذف از فصل:** این Test قبلاً مشاهده شده است و نباید برای نتیجه‌گیری درباره بهترین tau یا تأیید تعمیم تازه استفاده شود. نگهداری نسخهٔ قبلی و این جدول، امکان پیگیری تاریخی را حفظ می‌کند.

#### تفکیک از مطالعات مجاور

نسخهٔ پیشین §۴.۱۲ همچنین به چهار اجرای بازیابی مراکز دست‌یافتنی و هفت اجرای بازارزیابی مراکز ثابت اشاره داشت؛ **این ۱۱ اجرا جزو ۷۶ اجرای Benchmark نیستند** و فایل‌های اصلی و گزارش آن‌ها در `code/experiments/benchmark_v1/RESULTS_FA.md` و زیرپوشه `results/` باقی مانده‌اند. مباحث حل معکوس و کران دست‌یافتنی در بخش‌های قبلی فصل چهارم بررسی می‌شوند. حذف جزئیات تکراری آن‌ها از §۴.۱۲ حذف داده یا تغییر نتیجه نیست.

**محدوده و کنترل:** فقط نوشتار §۴.۱۲ و این README جدید، بدون train، optimizer، inference، Test، تولید/تغییر مرکز، دستکاری P1/P3، تغییر قرارداد، تغییر فصل ۳، ویرایش بخش‌های ۴.۱۰/۴.۱۱/۴.۱۳ و بدون ادغام.


## فهرست فنی پرونده‌ها و نقش هرکدام

| فایل | نقش در فرایند آزمایش / بازتولید |

|---|---|

| [`protocol/base_generator_config.json`](protocol/base_generator_config.json) | نتایج ماشین‌خوان یا پیکربندی |

| [`protocol/training_config_v1.json`](protocol/training_config_v1.json) | نتایج ماشین‌خوان یا پیکربندی |

| [`r03_center_recovery/boundary_fixture_results.json`](r03_center_recovery/boundary_fixture_results.json) | نتایج ماشین‌خوان یا پیکربندی |

| [`r03_center_recovery/center_only_rebuild.py`](r03_center_recovery/center_only_rebuild.py) | بازپخش/ممیزی خروجی‌های ثبت‌شده |

| [`r03_center_recovery/test_backfill_boundaries.py`](r03_center_recovery/test_backfill_boundaries.py) | آزمون صحت موتور/گزارش/دادهٔ ذخیره‌شده |

| [`results/archive_preflight.json`](results/archive_preflight.json) | مانیفست، گواه بازیابی یا اطلاعات منشأ |

| [`results/center_only_metrics.csv`](results/center_only_metrics.csv) | جدول اندازه‌گیری‌های عددی و هندسهٔ مرکز |

| [`results/center_only_rebuild_audit.json`](results/center_only_rebuild_audit.json) | مانیفست، گواه بازیابی یا اطلاعات منشأ |

| [`results/ci_provenance.json`](results/ci_provenance.json) | نتایج ماشین‌خوان یا پیکربندی |

| [`results/core_50_16runs.zip`](results/core_50_16runs.zip) | فایل خام آرشیو شامل checkpoint و اجرای ثبت‌شده؛ هویت SHA در پروتکل |

| [`results/coverage_manifest_76.json`](results/coverage_manifest_76.json) | مانیفست، گواه بازیابی یا اطلاعات منشأ |

| [`results/coverage_summary_76.csv`](results/coverage_summary_76.csv) | شاخص اجراها، شمارش یا خلاصهٔ عددی |

| [`results/extra_25_12runs.zip`](results/extra_25_12runs.zip) | فایل خام آرشیو شامل checkpoint و اجرای ثبت‌شده؛ هویت SHA در پروتکل |

| [`results/master_runs_76.csv`](results/master_runs_76.csv) | شاخص اجراها، شمارش یا خلاصهٔ عددی |

| [`results/posthoc_score_backfill_76_compact.csv`](results/posthoc_score_backfill_76_compact.csv) | شاخص اجراها، شمارش یا خلاصهٔ عددی |

| [`results/recovery_audit.json`](results/recovery_audit.json) | مانیفست، گواه بازیابی یا اطلاعات منشأ |

| [`results/size_25_24runs_seed20261102.zip`](results/size_25_24runs_seed20261102.zip) | فایل خام آرشیو شامل checkpoint و اجرای ثبت‌شده؛ هویت SHA در پروتکل |

| [`results/size_25_24runs_seeds20261103_20261104.zip`](results/size_25_24runs_seeds20261103_20261104.zip) | فایل خام آرشیو شامل checkpoint و اجرای ثبت‌شده؛ هویت SHA در پروتکل |

| `README_FA.md` | همین گزارش فنی علمی و راهنمای کامل پرونده‌های فعلی |

## سیاست بازیابی نسخه‌های تاریخی

Markdownهای گزارش/پروتکل قبلی در متن بالا تجمیع و از checkout برداشته شدند. نسخهٔ اصلی با شناسهٔ Git blob در [کامیت منبع](https://github.com/mahhammad/My-Thesis/tree/5ddce088204b422569c063ec6e22cb60a928c6c5/code/experiments/standard_policy_comparison/) باقی می‌ماند. دست‌زدن به فایل‌های داده، کد یا متن LaTeX خارج از دامنه این اقدام بوده است.
