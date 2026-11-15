# [RESEARCHER-03] CH4-METRICS-P1-PROVISIONAL-01 — Raw-only evidence audit

**وضعیت:** `READY_PENDING_QA76` — فقط بسته **موقت/مشروط**؛ نتیجه مستقل تأیید نشده و قابل انتشار قطعی در فصل چهارم نیست.  
**مرجع علمی ثابت:** `chapter4@de02fdda91e23db177de4429b3e1a9693c046d83`.  
**مبنای center-only ثابت:** `experiment/ch4-metrics-backfill-v2@0a552e6e31999f0a9aeab499f62d3b54313c2e1c`.  
**شاخه مستقل P1:** `experiment/ch4-metrics-p1-provisional-r03`؛ تنها فایل‌های جدید داخل `p1_provisional/`.  
**پروتکل قبلی:** [Issue #49 / GAP-PROTOCOL](https://github.com/mahhammad/My-Thesis/issues/49#issuecomment-6793559769).  
**رأی فعلی ممیز QA_76:** [`ACCEPT_WITH_LIMITATIONS`](https://github.com/mahhammad/My-Thesis/issues/49#issuecomment-6795313941) — گواه ۴۰ raw مرکز از جانب ممیز **هنوز وجود ندارد**.

## ۱) منابع، مرز مأموریت و گیت‌های SHA

برای دو ZIP تاریخی که رابط GitHub قبلاً بایت‌شان را خالی برمی‌گرداند، فایل‌های خام با نام‌های محلی `core_50_16runs(2).zip` و `size_25_24runs_seed20261102(1).zip` مستقیماً از کاربر دریافت و در Python خوانده شدند؛ نام تغییرکرده فایل هیچ اثری بر SHA ندارد:

| ZIP اصلی در مخزن | بایت | SHA-256 کاملِ فایل دریافت‌شده | عضو JSON |
|---|---:|---|---:|
| `core_50_16runs.zip` | 1,325,691 | `9609eb96c207316fde26934cbef4668ba287a275d4485175d1da7e798a297040` | ۱۶ |
| `size_25_24runs_seed20261102.zip` | 1,208,653 | `d570fb3f1dcab66476f027e1ade17b4e87c7ca16ff7fa6c76fda87b260c84e9f` | ۲۴ |

هر دو SHA **دقیقاً** با سند قفل‌شده GitHub برابر بودند؛ manifest هر ZIP، ۴۰ member SHA-256، `run_id/seed/family/objective/τ/steps/n`، `dataset.test_used=false`، تعداد reference IDs، شکل مرکز ۳×۳، positivity و نرمال‌شدن سطری نیز از **JSON خام** بررسی شد. از هیچ کلید Test به‌جز `test_used` برای گیت ممنوعیت استفاده نشده؛ هیچ Test sample / score خوانده نشده است.

۳۶ عضو دیگر از `../verified_sources_36.json` با blob `fe8ec364f9f2e9a00b3871234b177bda58dd512b` و provenance و member SHA تأییدشده قبلی خوانده شد. برای هر ۷۶ ردیف، association با `master_runs_76.csv` (blob `bcc2d7f4a4f39301973eb53c465ff05a6b4d0f21`) و `posthoc_score_backfill_76_compact.csv` (blob `5cea91d25db0602a30c350d39aada393f2592ac7`) بررسی شد؛ در جدول posthoc حداکثر تفاوت‌های طبیعی ناشی از گردکردن شش‌رقمی در BA/I/R_JS از `5e-7` فراتر نرفت.

**استدلال مثبت‌بودن مرکز:** کمینه entry برای ۴۰ مرکز ZIPهای ارسالی `0.014354832392577866` و برای ۳۶ مورد قبلی `0.024847449072930662` است. پس برای تمام ۷۶ مجموعه مراکز، support مرکز کامل است؛ ولی این **جای محاسبه عددی per-sample KL exact یا zero-support count نیست** و عدد جدیدی برای `R_KL_exact_support_aware` / `KL_candidate_support_zero_count` ساخته نشده است.

## ۲) تطبیق aggregate واقعی — نه graph forward و نه re-score

در هر عضوِ موجود، فقط `final.train`، `final.validation`، `per_class`، `counts`، `confusion_true_rows_pred_cols`، مراکز، متادیتای checkpoint/manifest و شناسهٔ اولیه مطالعه شد. **هیچ `raw_logits` برای استنتاج پردازش نشده**؛ تابع policy / engine فراخوانی نشده؛ center-only ۷۶تایی مجدداً محاسبه نشده است.

مقایسه مستقل و کم‌هزینه از **اعداد ذخیره‌شده** انجام شد: `BA=mean_c C_cc/n_c`، `ordinary_accuracy=ΣC_cc/Σn_c`، recall هر کلاس، `R_JS/R_sqrt_JS/R_KL_legacy_clipped = mean_c raw.per_class[c].R_*`، `mean_true_class_similarity=1−R_JS/ln2`، هویت محدود Train برای KL، کنترل ماتریس آشفتگی، و تطبیق BA/I و R تاریخی با manifest/master.

- **۴۰ ZIP جدید:** ۴۰ عضو با تمام SHA معتبر؛ **۶۰ رکورد aggregate split** (۴۰ Train + ۲۰ Validation)، ۱۲۰ مقایسه BA، ۶۰ مورد ordinary accuracy، ۱۸۰ recall و هرکدام ۶۰ مورد R_JS/R_sqrt_JS/R_KL تاریخی؛ **هیچ اختلاف در این معیارها فراتر از `1e-10`**. حداکثر باقیمانده BA حدود `2.220446049250313e-16` و در کل معیارهای هندسی نمونه‌ای بررسی‌شده حداکثر در حد گردکردن double بوده است.
- **۳۶ JSON قدیم:** ۳۶ Train + ۳۰ Validation + ۶ Validation نامعتبر؛ **۱۳۷۳ کنترل هویت/سازگاری سبک** در اجرای خواندن مرتبط، بدون خطا؛ بزرگ‌ترین اختلاف جمع‌بندی `2.0816681711721685e-16`. تمام member SHA با master برابر؛ ۶۶ مقایسه BA با posthoc در تحمل گردکردن گزارش ۶ رقمی است.
- **اختلاف ثبت‌شده و عمداً صاف‌نشده:** در هر ۴۰ JSON جدید، دو مقدار `run.elapsed_s` و `run.final.elapsed_s` کمی متفاوت‌اند؛ حداکثر اختلاف مطلق `0.0005203089999241683` ثانیه. این ۴۰ اختلاف در `raw_aggregate_audit_76.json` با شناسه run ثبت شده‌اند و **هیچ‌کدام به معنای اشتباه در BA/R یا سلامت SHA نیستند**. `experiment_specific.elapsed_s` از نظر فرمول/پروسه training همچنان historical است و عدد دلخواه جایگزین نشده است.
- `posthoc` برای **۷۶/۷۶ کلید مرکب** `(data_seed,n,family,objective,τ,steps)` به master متصل شد؛ اختلاف گردکردن BA، I و R_JS حدود `5e-7` یا کمتر بود، اما NLL/Softplus **با forward مستقل بازحساب نشده‌اند**.

## ۳) ماتریس کامل ۷۶×۵۵ و معناشناسی شش وضعیت

جدول `coverage_76x55_provisional.csv` شامل **۷۶ سطر**، متادیتای split، SHA هر member و برای **هر یک از ۵۵ metric ID** سه میدان `status/version/evidence` است؛ `metric_rules_55.json` نگاشت فیلد تاریخی/نسخه و شرط محاسبه را به‌ازای ۵۵ شناسه ثبت می‌کند. جدول شمارشی `coverage_summary_55_provisional.csv` به تفکیک ۵۵ شاخص تهیه شد.

| گروه | شناسه | `measured` | `recomputed_from_raw` | `historical_unverified` | `not_measured` | `not_applicable` |
|---|---:|---:|---:|---:|---:|---:|
| مراکز | ۱۱ | ۸۳۶ | ۰ | ۰ | ۰ | ۰ |
| Train | ۲۱ | ۰ | ۵۳۲ | ۶۸۴ | ۳۸۰ | ۰ |
| Validation | ۲۱ | ۰ | ۳۵۰ | ۴۵۰ | ۲۵۰ | ۵۴۶ |
| آزمایش خاص | ۲ | ۰ | ۰ | ۱۵۲ | ۰ | ۰ |
| **مجموع** | **۵۵** | **۸۳۶** | **۸۸۲** | **۱۲۸۶** | **۶۳۰** | **۵۴۶** |

`undefined_or_infinite=0` **وضعیت ثبت‌شدهٔ فعلی** است؛ نبود مقدار برآوردشده برای KL exact، به‌معنای اثبات متناهی‌بودن تمام NLLها نیست. تعداد کل خانه‌ها **۴۱۸۰ = ۵۵×۷۶**. `pairwise_JS_similarity` مشتق کمکی است و شناسهٔ ۵۶ام محسوب نشده است.

**تعریف دقیق سطوح:**
- `measured`: فقط ۱۱ معیار center-only که قبلاً از raw محاسبه و در candidate گزارش شدند؛ **QA_76 برای ۴۰ مورد هنوز مستقل قبول نشده است**.
- `recomputed_from_raw`: **صرفاً جبر جمع‌بندی ذخیره‌شده** مانند میانگین مساوی per-class و Confusion، نه اجرای دوباره تابع JS/KL روی (P_i). این وضعیت برای Train/Validation در (R_{JS})، (R_{\sqrt{JS}})، شباهت کلاس درست، BA، accuracy عادی، recall کلاس و (R_{KL}^{\rm legacy}\) به کار رفته است؛ **اعتبار کامل v2 از آن استنتاج نشود**.
- `historical_unverified`: مقدار **واقعاً ذخیره‌شده** ولی نه محاسبه مستقل از per-sample (از جمله (R_{KL}\) اصلی، confusion، margin quantiles، hard-margin، sigmoid، NLL و Softplus). `experiment_specific.elapsed_s` و objective loss نیز historical باقی می‌مانند.
- `not_measured`: نبود per-sample score/representation برای `tie_count`، `exact_tie_count`، `near_tie_count_atol_1e-12`، `R_KL_exact_support_aware`، و `KL_candidate_support_zero_count`.
- `not_applicable`: دقیقاً **۲۶ رکورد** از `node/node_class` برای **هر یک از ۲۱ معیار Validation**، چون query-start برای unseen sample تعریف نشده؛ خروجی `final.validation` خام نیز وضعیت `not_applicable` دارد.
- `undefined_or_infinite`: فقط با شاهد ریاضی واقعی از per-sample/support ثبت می‌شود، نه به‌عنوان جایگزین missing.

**سه تمایز اصلی ریاضی:** `R_KL` قراردادی باید exact support-aware باشد، در مقابل `R_KL_legacy_clipped` که `clamp_min(1e-300)` دارد. `R_sqrt_JS` میانگین sqrt در سطح هر نمونه و سپس کلاس است و با `sqrt(R_JS)` یکی نیست. `tie_count` گزارشگر `report_snapshot.py` براساس `isclose(atol=1e-12,rtol=0)` near-tie است، نه exact-tie؛ بدون scoreهای per-sample از یکدیگر یا از BA قابل استنتاج نیستند. **`training_tau` هدف بهینه‌سازی را با `posthoc_softplus_tau=0.005` یکسان نخوانید.**

## ۴) مرزهای نبود شاهد و گام‌های نیازمند P2

Checkpointهای تاریخی `raw_logits` را دارند، ولی خروجی raw summary فعلی حاوی **تمام `P_i^(c)` و `S_c(X_i)` تک‌نمونه‌ای نیست**. برای داوری واقعی score، tie، exact-support KL و NLL نسخه v2 نیاز به **مجوز جداگانه P2 inference-only** با قفل Train/Validation، سیاست query-start معتبر، guard جلوگیری از Test و regression against original checkpoints است. این پروتکل **هیچ forward** انجام نداد؛ صرفاً مرز مدرک و نیاز را تعیین کرد. در `node/node_class` بدون start policy معتبر، Validation نه در P1 و نه خودکار در P2 قابل اصلاح نیست.

## ۵) رأی مستقل، fail-closed و منابع

**مسئله QA مستقل:** رأی مستقل فعلی [Issue #49 / 6795313941](https://github.com/mahhammad/My-Thesis/issues/49#issuecomment-6795313941) = `ACCEPT_WITH_LIMITATIONS` برای ۳۶ raw قابل استخراج، نه `ACCEPT_QA76_CENTER_ONLY`. ممیز در گفت‌وگوی خودش به دو ZIP بزرگ دسترسی بایتی نداشت؛ این دو **فقط در گفت‌وگوی پژوهشگر ۳ آپلود شدند**. کنترل SHA و aggregate پژوهشگر ۳ **به‌هیچ‌وجه جای رأی مستقل** برای ۴۰ مرکز باقی‌مانده را نمی‌گیرد.

**گیت پذیرش:** ۱) ممیز مستقل SHA کامل دو ZIP و member raw و ۴۰ ماتریس ۳×۳ را دریافت و با قاعده مرکز canonical و ۱۱ معیار از نو بررسی کند؛ ۲) برای `ACCEPT_QA76_CENTER_ONLY`، ۷۶ مرکز قابل قبول می‌شوند و سپس ممیز/مدیر P1 را با raw و قرارداد metric status جدا review کنند؛ ۳) حتی پس از پذیرش QA_76، هیچ `historical_unverified` یا `not_measured` در P1 خودکار به `verified_v2` تغییر نکند. **اگر QA_76 رد یا اختلاف SHA مشاهده شد، انتشار P1 متوقف/قرنطینه و تطبیق منشأ با اجازه جدید انجام شود؛ نه اصلاح بی‌صدا**. اگر فقط قبول مشروط بود، وضعیت P1 نیز candidate می‌ماند. **هیچ claim فصل چهارم یا ادغام خودکار مجاز نیست.**

**منابع و بودجه:** دو ZIP ارسالی با Python standard library و `zipfile/json/hashlib` ممیزی شدند؛ پردازش اصلی ۴۰ عضو در اجرای ثبت‌شده **۰٫۲۴۰۲ s** wall عددی و **۱۰۲٫۰۳ MiB** max RSS (Linux `ru_maxrss`) داشت. دو اجرای اولیه fail-closed به سبب تشخیص placeholder `not_applicable` و اختلاف timestamp ثبت/رفع شدند؛ جمع مرحله‌های عددی بسیار پایین‌تر از ۱۸۰ s. کنترل‌های JS روی ۳۶ مورد با GitHub connector انجام شد؛ **ابزار connector امکان ثبت مستقل RSS همان runtime را نمی‌دهد، بنابراین RSS مجموع این محیط `NOT_MEASURED` ثبت می‌شود** و انطباق اندازه‌گیری‌شده برای آن ادعا نمی‌شود. هیچ dependency جدید، CI، Test، graph forward، بازآموزی، بازپخش، تغییر قرارداد یا LaTeX انجام نشد.

**رأی پژوهشگر ۳:** `READY_PENDING_QA76` — تکمیل P1 با منشأ قابل‌ردیابی و ۴۰ discrepancy زمان، **provisional only**.

**RESEARCHER-03 — پایان مأموریت و توقف تا تصمیم مدیر پژوهش.**
