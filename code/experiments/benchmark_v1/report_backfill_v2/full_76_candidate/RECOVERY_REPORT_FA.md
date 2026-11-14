# [RESEARCHER-03] CH4-METRICS-ARCHIVE-RECOVERY-01 — گزارش بازیابی مرکزهای ۷۶ اجرا

**عامل:** پژوهشگر شماره ۳ — RESEARCHER-03  
**تاریخ:** ۱۱ نوامبر ۲۰۲۶  
**منبع قفل‌شده:** `chapter4@de02fdda91e23db177de4429b3e1a9693c046d83`  
**شاخهٔ صرفاً پژوهشی:** `experiment/ch4-metrics-backfill-v2`  
**قرارداد:** `chapter4-common-evaluation-contract-v2`  
**رأی پیشنهادی:** `READY_FOR_INDEPENDENT_QA_76`، **محدود به center-only**، نه پوشش کامل تمام معیارهای نمونه‌ای.

## ۱. بازیابی و کنترل تمام آرشیوها

در checkout واقعیِ private GitHub repository و پیش از محاسبه، نسخه کد مرجع `code/policy_evaluator/metrics.py`، جدول `master_runs_76.csv`، قرارداد مشترک و blob SHA آرشیوها با commit قفل‌شده مطابقت داده شدند؛ `center_only_rebuild.py` و خروجی‌های ۳۶ مورد اولیه با `868a3167413a9590a581fc1d2a682969fedeff9e` قفل شدند. `preflight_recovery.py` برای هر ZIP **هش بایت به بایت SHA-256**، شمار اعضا، هش هر عضو علیه مانیفست داخلی و master CSV، هویت `run_id/seed/family/objective/steps` و `test_used=false` را بررسی کرد. نتیجه:

| منبع ZIP | اعضای معتبر | SHA-256 |
|---|---:|---|
| `core_50_16runs.zip` | ۱۶ | `9609eb96c207316fde26934cbef4668ba287a275d4485175d1da7e798a297040` |
| `extra_25_12runs.zip` | ۱۲ | `061a00923ae69ac2f8b923947bd9cdea9d4f3dba1d1771063bcb3ecffc0201e3` |
| `size_25_24runs_seed20261102.zip` | ۲۴ | `d570fb3f1dcab66476f027e1ade17b4e87c7ca16ff7fa6c76fda87b260c84e9f` |
| `size_25_24runs_seeds20261103_20261104.zip` | ۲۴ | `fdb862462e237e5094204ff17a5bc1a79a0f1d17f3a9b9d90ce967dba8b11fdd` |

**۷۶/۷۶ عضو یکتا** کنترل شدند. نسبت خانواده‌ها: `shared=25`, `class=25`, `node=13`, `node_class=13`. دقت Validation برای ۵۰ اجرای shared/class قبلاً ثبت شده؛ برای ۲۶ اجرای node/node_class بدون شروع query، `not_applicable` باقی مانده است. بذرها، گام‌های ۲۵/۵۰ و ماهیت posthoc در `coverage_manifest_76.json` جدا باقی می‌مانند.

## ۲. محاسبه فقط از مرکز — ۷۶/۷۶

پس از موفقیت کنترل همه آرشیوها، **اسکریپت موجود** `center_only_rebuild.py --execute` اجرا شد؛ این اسکریپت مستقیماً از تابع مرجع فصل سوم در `code/policy_evaluator/metrics.py::center_metrics` استفاده کرده است. خروجی `center_only_metrics.csv` شامل دقیقاً ۷۶ ردیف محاسبات `T, BT, TNL, BTNL, I`، ماتریس زوجی `JS/√JS`، شباهت JS، میانگین و کمینه جدایی، و نزدیک‌ترین فاصله هر مرکز است. هیچ سیاستی دوباره آموزش ندید و هیچ گشتی بازپخش نشد. **۳۶ ردیف فایل قبلی `../center_only_metrics.csv` و `../verified_sources_36.json` دست‌نخورده مانده‌اند.**

مقایسه عدد به عدد **تمام فیلدهای مرکز** در ۳۶ ردیف قدیمی با خروجی ۷۶تایی از آستانه `1e-10` گذشت. بیشترین اختلاف `8.604228440844963e-16` و برای `I` حداکثر `4.440892098500626e-16` بود؛ اختلاف فراتر از تحمل: **صفر**.

**تمایز علمی:** `R_KL, R_JS, R_sqrt_JS`، BA، حاشیه، NLL و tie از توزیع‌های نمونه‌ای وابسته‌اند؛ در این مرحله **دوباره محاسبه نشده‌اند**. اعداد این گروه‌ها در پوشش‌نامه به‌عنوان شواهد **legacy/posthoc** با سطح منشأ تفکیک‌شده نگهداری شده‌اند. تکیه‌گاه KL برای ۴۰ مورد جدید با محاسبه sample-level بررسی نشده است؛ `R_KL` اصلی قرارداد v2 برای آن‌ها همچنان `not_verified` است، حتی اگر عضو ZIP اکنون SHA-verified باشد. گزارش کامل معیارهای v2 هنوز موضوع ممیزی/تصمیم جداگانه خواهد بود.

## ۳. فایل‌های خام، پوشش و قابلیت ممیزی

- [archive_preflight.json](archive_preflight.json) — SHA هر چهار ZIP، ۷۶ member SHA، هویت و نوع split.
- [center_only_metrics.csv](center_only_metrics.csv) — ۷۶ ردیف معیارهای مرکز؛ گواه اصلی تکمیل این مأموریت.
- [center_only_rebuild_audit.json](center_only_rebuild_audit.json) — اجرای اسکریپت canon، ۷۶ عضو، بدون اختلاف فراتر از تحمل.
- [recovery_audit.json](recovery_audit.json) — مقایسه ۳۶ قبلی، آستانه و بیشینه اختلاف هر فیلد.
- [coverage_manifest_76.json](coverage_manifest_76.json) و [coverage_manifest_76.csv](coverage_manifest_76.csv) — **۷۶ run_id و ۵۵ metric ID** با وضعیت `measured/not_measured/not_applicable/undefined_or_infinite/not_verified` و دلیل؛ archive/member SHA و `source_evidence` از `metric_status` جداست.
- [coverage_summary_76.csv](coverage_summary_76.csv) — پوشش شمارشی ۵۵ metric؛ هر ۱۱ معیار الزامی center-only: **۷۶ measured، صفر not_verified**.
- [recovery_budget.json](recovery_budget.json) و [ci_provenance.json](ci_provenance.json) — حسابرسی RSS/زمان و نسخه محیط.

## ۴. تست‌ها، CI و محدودیت منابع

**CI موفق:** GitHub Actions workflow `RESEARCHER-03 archive recovery — bounded center-only 76 candidate`، [Run 42998645098](https://github.com/mahhammad/My-Thesis/actions/runs/42998645098)، روی نسخه `a3af4749e94612e56b9bc0d5445bec0d3556dd97`، دو job مستقل را PASS کرد.

- **N00 boundaries:** ۵ تست importشده از مخزن PASS؛ Python `3.11.18`، NumPy `2.4.6` و PyTorch CPU `2.15.0+cpu`. عددگیری و unittest: **۱٫۵۰۶۳ ثانیه**، اوج RSS مجموع درخت پردازش: **۳۱۸٫۵۴ MiB**.
- **Archive recovery:** Python `3.11.18`، NumPy `2.4.6`، بدون نصب Torch در مرحله center. کنترل SHA + محاسبه مرکز + تطبیق ۳۶ نتیجه: **۰٫۸۰۶۴ ثانیه**، اوج RSS درخت: **۱۰۲٫۹۳ MiB**.
- مجموع مراحل عددی اجرای موفق حدود **۲٫۳۱۳ ثانیه**، بسیار پایین‌تر از ۱۸۰ ثانیه؛ سقف ۱ GiB رعایت شد. نصب dependencies خارج از سنجش عددی و در CI ثبت شده است. runner از timeout و guard نمونه‌برداری RSS روی process tree و ثبت JSON استفاده کرد.
- **تاریخچه شفاف اصلاح گزارشگر:** دو اجرای پیشین [42998193781](https://github.com/mahhammad/My-Thesis/actions/runs/42998193781) و [42998419218](https://github.com/mahhammad/My-Thesis/actions/runs/42998419218) در **لایه تبدیل شاخص کمکی `pairwise_JS_similarity` به قرارداد پوشش** fail-closed متوقف شدند. در هر دو اجرا همه SHAها و محاسبات ۷۶ مرکز موفق بودند اما **هیچ candidate ناقصی روی شاخه publish نشد**. اصلاح فقط در `verify_recovery.py` بود؛ `center_only_rebuild.py`، موتور گشت و فرمول‌های مرکز عوض نشدند. این دو attempt نیز زیر سقف‌های numerical/RSS بودند؛ اجرای سوم کامل و قابل ممیزی شد.

## ۵. گیت و پیشنهاد اقدام بعدی

**رأی مجری: `READY_FOR_INDEPENDENT_QA_76` فقط برای center-only.** ممیز مستقل باید checksum هر چهار ZIP/مانیفست، تمایز ۵۵ معیار رسمی از `pairwise_JS_similarity` کمکی، تمام ۷۶ مرکز و اختلاف ≤`1e-10` با ۳۶ نتیجه پیشین را کنترل کند؛ `coverage_manifest_76.json` قابل audit است. تا رأی ممیز مستقل هیچ PR/merge یا انتشار ادعای «۷۶ گزارش کامل v2» مجاز نیست.

**نه Test، نه آموزش، نه بازپخش استنتاج، نه تغییر LaTeX، نه overwrite آرشیوها یا خروجی اولیه.**

**RESEARCHER-03 — پژوهشگر شمارهٔ ۳**
