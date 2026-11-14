# [RESEARCHER-03][P3-CLAIMS] CH4-METRICS-P3-PUBLICATION-GATES-01

**مأموریت:** نقشهٔ ادعاهای قابل نگارش، گیت‌های انتشار و حداقل برنامهٔ احتمالی P2  
**عامل:** پژوهشگر شمارهٔ ۳ (`RESEARCHER-03`)  
**رأی پیشنهادی:** `READY_FOR_PUBLICATION_GATE_REVIEW` — صرفاً آمادهٔ بررسی مدیریت، **نه READY_FOR_PUBLICATION یا verified-v2**  
**تاریخ:** ۱۱ نوامبر ۲۰۲۶  
**مرجع علمی قفل‌شده:** `chapter4@de02fdda91e23db177de4429b3e1a9693c046d83`  
**P1 قفل‌شده:** `experiment/ch4-metrics-p1-provisional-r03@d36ec6dc522e4e6d5f7ee4134f23c71f97b2aff3`  
**وضعیت مدیریتی P1:** [`ACCEPT_P1_DELIVERY_PENDING_QA76`](https://github.com/mahhammad/My-Thesis/issues/49#issuecomment-6795533245)؛ **رأی مستقل فعلی QA76:** [`ACCEPT_WITH_LIMITATIONS`](https://github.com/mahhammad/My-Thesis/issues/49#issuecomment-6795313941). تکلیف ۴۰ مرکز در [`V-CH4-METRICS-RAW40-SUPPLEMENT-01`](https://github.com/mahhammad/My-Thesis/issues/49#issuecomment-6795448684) جداگانه پیگیری می‌شود. **هیچ رأی `ACCEPT_QA76_CENTER_ONLY` در مبنای این گزارش وجود ندارد.**

## ۱. صورت‌بندی مسئله: آنچه داریم در برابر آنچه قابل ادعای علمی است

ماتریس P1 شامل **۷۶ checkpoint، ۵۵ شناسهٔ وضعیت و ۴۱۸۰ خانه** است؛ ۱۱ شناسه به هندسهٔ مراکز اختصاص دارند. حساب وضعیت P1 (یعنی وضعیت شواهد، نه تصویب اعتبار علمی) چنین است:

| سطح شاهد P1 | تعداد خانه | تفسیر مجاز |
|---|---:|---|
| `measured` | ۸۳۶ | تمام ۱۱ سنجهٔ مرکز در ۷۶ اجرا، **candidate در انتظار QA مستقل ۴۰ مرکز** |
| `recomputed_from_raw` | ۸۸۲ | فقط حساب aggregate روی summaryهای تاریخی خام؛ نه بازتولید per-sample |
| `historical_unverified` | ۱۲۸۶ | عدد ثبت‌شدهٔ تاریخی/پساارزیابی؛ بدون اعتبارسنجی مستقل v2 |
| `not_measured` | ۶۳۰ | نبود بازنمایی/score تک‌نمونه‌ای برای سنجهٔ خواسته‌شده |
| `not_applicable` | ۵۴۶ | ۲۶ خانوادهٔ آزاد گره‌ای × ۲۱ سنجهٔ Validation بدون قرارداد شروع query |
| `undefined_or_infinite` | ۰ | **فقط مقدار وضعیت P1**؛ اثبات متناهی‌بودن همه سنجه‌های محاسبه‌نشده نیست |

**تفاوت کلیدی:** `measured != independently_verified` و `recomputed_from_raw != per_sample_recomputed`. **۴۱۸۰ سلول وضعیت، ۴۱۸۰ مقدار عددی مستقل نیستند.** `pairwise_JS_similarity` شاخص کمکی است و شناسهٔ ۵۶ام قرارداد نیست.

### جدول تصمیم ۵۵ معیار × چهار خانواده

[CLAIM_GATE_MATRIX_55x4.csv](CLAIM_GATE_MATRIX_55x4.csv) دقیقاً **۲۲۰ سطر** دارد: یک سطر برای هر جفت `(metric_id,family)` و ستون‌های شمار ۶ وضعیت P1، نسخه metric، مبنای مجاز، SHA/فایل، مانع گیت، عبارت امن، جایگاه در فصل چهارم و ورودی حداقلی P2. گروه‌ها: `shared=25`, `class=25`, `node=13`, `node_class=13`. این جدول ادعای آماری روی ۲۲۰ تکرار جدید نمی‌کند؛ هر سطر حکم انتشار یک سنجه در یک خانواده است.

| حالت تصمیم (یک مقدار برای هر metric×family) | سطر |
|---|---:|
| `REPORTABLE_NOW_AS_HISTORICAL` | ۹۰ |
| `QA76_GATED` | ۴۴ |
| `P2_INFERENCE_REQUIRED` | ۳۶ |
| `PERMANENT_NOT_APPLICABLE_WITHOUT_NEW_QUERY_START_CONTRACT` | ۴۲ |
| `OPTIONAL` | ۸ |
| **مجموع** | **۲۲۰** |

- `REPORTABLE_NOW_AS_HISTORICAL` یعنی **نوشتن توصیفی به نام نسخه تاریخی با ارجاع source** مجاز است، نه امضای انتشار نهایی v2؛ ادعاهای چندمعیارهٔ همراه مراکز همچنان مشروط به QA76 می‌شوند.
- `QA76_GATED` برای **۱۱ سنجهٔ مرکز × ۴ خانواده** است؛ ۳۶ مرکز مستقل ممیزی شده، اما برای ۴۰ باقی‌مانده رأی کامل صادر نشده است.
- `P2_INFERENCE_REQUIRED` برای ادعای **exact v2** از جمله `R_KL` (با وجود scalar قدیمی clipped)، `R_KL_exact_support_aware`، support-count و tieهاست. بعضی نسخه‌های تاریخی را **می‌توان با نام دقیق تاریخی** توصیف کرد، ولی این طبقه‌بندی حق چاپشان به‌عنوان v2 را نمی‌دهد.
- `PERMANENT_NOT_APPLICABLE_WITHOUT_NEW_QUERY_START_CONTRACT` دربارهٔ همان **۲۶ رکورد node/node_class** در Validation است؛ «permanent» به معنی نامربوط **در پروتکل قفل‌شدهٔ موجود** است، نه ادعای ناممکن‌بودن پژوهش آینده با سیاست جدید. ۲۱ سنجهٔ Validation × دو خانواده = **۴۲ سطر تصمیم**.
- `OPTIONAL` برای `elapsed_s` و `optimizer_objective_loss` (هر کدام چهار خانواده)، با برچسب provenance/هزینه و عدم مقایسهٔ objectives نامتجانس.

## ۲. Ledger ادعا و شیوه نگارش امن — اولویت‌های P0/P1/P2

| ادعا/مثال | شاهد قابل پذیرش اکنون | گیت مسدودکنندهٔ ادعای قوی‌تر | عبارت مجاز در متن | محل پیشنهادی فصل چهارم | اولویت |
|---|---|---|---|---|---|
| شمار بایگانی: ۷۶ `run_id` از چهار خانواده | `master_runs_76.csv@bcc2d7f4...`، SHA اعضا و `P1`؛ ممیز هویت جدولی ۷۶ را پذیرفته | تکرارهای هم‌بذر و زیرمجموعهٔ تودرتو، مستقل آماری نیستند | «۷۶ اجرای بایگانی‌شده با خانواده/هدف/بودجهٔ مشخص» | پروتکل داده و طرح Benchmark | **P0** |
| هندسهٔ مرکز، مثال `center.I`, `T/BT/TNL/BTNL`, فاصلهٔ JS | `center_only_metrics.csv@4e7ab942...`، رأی ممیز ۱۱×۳۶؛ مابقی candidate | **`ACCEPT_QA76_CENTER_ONLY`** برای ۴۰ raw مرکز | «مقادیر ۷۶ مرکز از checkpointها محاسبه و در انتظار پذیرش مستقل کامل است» | جدول هندسه مراکز / پیوست | **P1** |
| Train، مثال `train.R_JS` و `train.R_sqrt_JS` | `final.train.per_class` خام و گزارش P1 aggregate؛ ۷۶ حساب داخلی | نسخهٔ JS per-sample را independent v2 نمی‌نامیم | «دو معیار تاریخی تراکم ذخیره و سازگاری تجمیعی‌شان کنترل شد» | جدول Train با نسخه معیار | **P0** |
| Train، `balanced_accuracy` و `ordinary_accuracy` | confusion/labels summaries، با حساب macro/micro از raw | حساسیت به بذر و بودجه؛ نبود generalization در node-family | «BA تاریخی روی Train ثبت و از ماتریس آشفتگی بازجمع شده است» | جدول‌های عملکرد Train | **P0** |
| Validation، ۵۰ shared/class | historical `final.validation` و `master_runs_76.csv`، P1 | پیش‌بینی بر پایه label واقعی query نباید انجام شده باشد؛ ارزیابی کامل v2 مجزاست | «Validation تاریخی برای ۵۰ اجرای دارای policy شروع معتبر گزارش شد» | جدول Validation / توضیح held-out | **P0** |
| Validation، ۲۶ `node/node_class` | `validation_status=not_defined_without_F` و P1 با **۵۴۶ خانه N/A** | تنها **قرارداد query-start جدید + آزمایش جدید** | «برای این ۲۶ اجرا معیارهای Validation تعریف‌ناپذیرند، نه صفر یا شکست» | پانویس جدول Validation / محدودیت‌ها | **P0** |
| `R_KL` v2 exact و حمایت توزیعی | `R_KL_legacy_clipped` تاریخی، مرکزهای مثبتِ گزارش‌شده؛ **نه** نمونه P | P2 per-sample `P_i^(c)` و کنترل support، تمایز infinite | «KL تاریخی با clipping محاسبه شده و KL دقیق مستقل بازامتیازدهی نشده» | تحلیل تراکم/محدودیت سنجه‌ها | **P2** |
| JS-similarity NLL / Softplus و NLL-KL تاریخی | `posthoc_score_backfill_76_compact.csv@5cea91d2...` با گردکردن شش‌رقمی | P2 برای ادعای verified-v2 و نسخه/دما؛ JS logits posterior Bayes نیست | «سنجه‌های NLL/Softplus posthoc و نسخه‌دار گزارش شده‌اند» | مقایسه اهداف / پیوست | **P0** (historical)، **P2** (v2) |
| `exact_tie_count` و `near_tie_count_atol_1e-12` | **هیچ score تک‌نمونه‌ای موجود در P1** | P2 نگهداری/بازبینی تمام candidate scores | «در دادهٔ تجمیعی فعلی تعداد tie دقیق/نزدیک اندازه‌گیری نشده است» | محدودیت طبقه‌بندی | **P2** |
| رابطهٔ `I` با BA یا ادعای برتری سیاست | مرکز candidate + BA تاریخی در همان `run_id` | QA76 + کنترل هم‌بودجه/بذر؛ علیت با داده موجود ثابت نمی‌شود | «هم‌تغییری توصیفی در اجراهای ثبت‌شده بررسی می‌شود؛ نتیجهٔ علی استنباط نمی‌شود» | بحث نتایج و محدودیت‌ها | **P1** |
| زمان اجرای آزمایش | `run.elapsed_s` و `final.walltime_s`، P1 discrepancy ledger | تعریف دقیق field و کنترل منابع و محیط؛ **نه** bug قطعی | «زمان کل ثبت‌شدهٔ اجرای الگوریتم از checkpoint interval جدا گزارش می‌شود» | مشخصات محاسباتی / پیوست | **P0** |
| `optimizer_objective_loss` | `final.objective_loss` و manifest/master | مقیاس/تعریف objectiveها متفاوت‌اند، هیچ training جدیدی انجام نشد | «زیان هدفِ مختص هر اجرا ثبت شده؛ مقایسه فقط در هدف و بودجهٔ یکسان» | روش آموزش / فراداده | **P0** |
| عدم‌قطعیت ناشی از اندازهٔ Train و بذر | archive metadata، seedهای `20261102/03/04`، overlap اندازه‌ها | سه seed و زیرمجموعهٔ هم‌پوشان کفایت آزمون قطعی را ندارند | «آمار تکرارها توصیفی و وابستگی داده‌ها صریح است» | بحث اعتبار بیرونی | **P0** |

**قاعده واژگانی:** برای `historical_unverified` از «ثبت شده/تاریخی» استفاده شود، نه «تأیید شده». برای `recomputed_from_raw` نوشته شود «حساب جبر جمع‌بندی ذخیره‌شده»، نه «محاسبه مجدد metric از P». برای `center` تا QA کامل «candidate»، نه «independently validated 76». **Test تاریخی صرفاً provenance است؛ در انتخاب مدل/آزمون فرضیه/پیشنهاد P2 استفاده نشد.**

## ۳. متن فصل چهارم و مستندکردن شواهد

[CH4_DRAFT_FA.md](CH4_DRAFT_FA.md) یک متن کوتاه و دانشگاهیِ قابل بازبینی شامل بند **معرفی Benchmark**، بند **پوشش و محدودیت Validation**، بند **پیوست الکترونیکی** و پانویس جدول‌های center/Train/Validation/NLL/time ارائه می‌کند. متن، معادلات فصل سوم را **تکرار نمی‌کند** و برای سنجه‌ها به تعاریف فصل سوم ارجاع می‌دهد. هیچ تغییر `tex/` انجام نشده است. **این پیشنویس مجوز جایگذاری نهایی نیست.**

**نقشه نگهداری الکترونیکی:** چهار ZIP تاریخی و `manifest.json`ها باید immutable بمانند. پیوند هر claim به `run_id`، `source_member_sha256`، blob SHA جدول، `metric_version`، `split` و `status` ثبت شود. خروجی center-only و P1 را در پوشه‌های نسخه‌دار جدا نگه دارید؛ در پیوست اصلی به جای تکرار فرمول، تعریف منشأ، نحوهٔ محاسبه و آنچه **محاسبه نشده** را توضیح دهید. `posthoc` شش‌رقمی **verified-v2** نیست.

## ۴. بررسی دقیق اختلاف زمان — دو timestamp با معنای متفاوت

در کد مرجع [`replay_v1.py`](https://github.com/mahhammad/My-Thesis/blob/de02fdda91e23db177de4429b3e1a9693c046d83/code/experiments/benchmark_v1/replay_v1.py) به blob `4d9e15abb37fb25ab531f36ad1cca93ad922e6a3`، `fit` در خط ۱۰۲ `start=time.perf_counter()` می‌گذارد:
- **`run.final.walltime_s`** از **`history[-1].walltime_s`** است؛ در خط ۱۱۴ `round(time.perf_counter()-start,4)` هنگام ثبت آخرین checkpoint اجرا می‌شود. این عدد **زودتر** و **با دقت چهار رقم اعشار** ثبت می‌شود.
- **`run.elapsed_s`** در خط ۱۲۵ مجدداً از `perf_counter` خوانده می‌شود، پس از ساختن `final` و فراخوانی `z.detach().tolist()` در خط ۱۲۴. این عدد **دیرتر و با دقت اصلی float** ثبت می‌شود؛ manifest خط ۱۸۳ همین `run.elapsed_s` را منتقل می‌کند. این مدت، زمان کلِ مسیر اجرا از نقطهٔ `start` تا انتهای آماده‌سازی خروجی است (شامل scoring دوره‌ای و برخی ساختارهای خروجی)، **نه زمان خالص گام‌های optimizer و نه زمان نوشتن ZIP روی دیسک**.

در گزارش P1 به‌اختصار از عبارت «`final.elapsed_s`» استفاده شده که **با کلید واقعیِ نشان‌داده‌شده در اسکریپت مرجع یکی نیست**؛ نام درستِ مسیر در مدل این کد **`final.walltime_s`** است. این اصلاح **فقط توضیح/نام‌گذاری در سند P3** است؛ آرشیو و جدول P1 تغییر داده نشده. اختلاف غیرصفرِ ۴۰ مورد با بیشینه `0.0005203089999241683 s` با نقطه‌های زمان‌گیری متفاوت و گردکردن سازگار است؛ **اثبات bug نیست**. البته نسبت‌دادن علت قطعی به هر ۴۰ عضو مستلزم ممیزی provenance دقیق نسخهٔ runner اصلی است.

**سیاست انتشار:** زمان اصلی را **`run.elapsed_s` از raw checkpoint / manifest** با عنوان «مدت ثبت‌شده از start تا آماده‌شدن خروجی» و ذکر محیط/بذر/گام گزارش کنید؛ `final.walltime_s` را با عنوان «زمان آخرین checkpoint history (rounded 4 decimals)» در metadata حفظ کنید. **هر دو را جداگانه و بدون averaging/overwrite** نگه دارید. اختلاف‌های ۴۰تایی در `p1_provisional/raw_aggregate_audit_76.json` محفوظ است؛ اختلاف کوچک به‌تنهایی سبب رد متریک‌ها یا سالم‌بودن قطعی ابزار time نیست.

## ۵. P2 آینده: کم‌هزینه‌ترین پاسخ به شکاف‌های واقعاً ضروری

[P2_MINIMAL_DECISION.md](P2_MINIMAL_DECISION.md) یک پروتکل **صرفاً پیشنهادی** است؛ نوبت اول **دو checkpoint از `shared/class` با `data_seed=20261102,n_ref=50,objective=margin_JSD,training_tau=.005,steps=25`**، هردو با Train+Validation. فقط اگر تصمیم مدیریتی شواهد Train گره‌ای را ضروری دانست، مرحلهٔ بعد **دو checkpoint از `node/node_class` فقط روی Train** با همان بذر/اندازه/هدف و گام است. SHA عضو هر چهار مورد، archive و کد قفل‌شده در سند P2 ثبت شده‌اند؛ **هیچ انتخاب پسینی بر اساس BA/Test انجام نشده است**.

برای exact-KL و `zero-support`: `P_i^{(y)}` یا تمام `P_i^{(c)}` همراه center و class label؛ برای exact/near tie، margin، JS-NLL، Softplus و sigmoid: **تمام `S_{i,c}` تک‌نمونه‌ای** و version/tau. برای KL-based NLL نیز همه candidate distributions لازم است. این داده‌ها در P1 محفوظ نشده‌اند؛ historical scalar یا مرکز تنها کافی نیستند. **هیچ Validation برای node/node_class بدون قرارداد query-start جدید اجرا نشود**. Pilot حداکثر یک گواه امکان‌سنجی نسخه‌ای روی چند عضو است، نه اصلاح خودکار ۷۶×۵۵.

**پیش از P2:** PASS مستقل QA76، مأموریت جداگانه با حد بودجه مصوب، قفل source/seed/reference IDs، تست N00 و fingerprints، عدم Test، توقف در mismatch/RSS/wall. تخمین اولیه برای مرحلهٔ A: **۶۰–۱۸۰ ثانیه wall**، **RSS≤۱ GiB** و **خروجی فشرده <۲ MiB**؛ **این‌ها تخمین نه budget مصوب و نه زمان اجراشده‌اند.** مرحله B هم مستقل تصویب شود. مرحله کامل ۷۶ اجرا صرفاً با اثبات نیاز علمی و گیت هزینهٔ جدید مطرح شود.

## ۶. گیت نهایی انتشار و fail-closed در برابر رد QA76

**P0 — نوشتن مشروط از شواهد حاضر:** مقدمهٔ پروتکل، پوشش archive، نسخه‌های معیار، BA/R_JS/R_sqrt_JS **تاریخی با برچسب روشن**، وضعیت N/A و محدودیت بذرها؛ بدون اعلام تأیید v2. **تمام جداول تاریخی همراه SHA و نوع aggregate** باشند.

**P1 — ممیزی مستقل RAW40، بدون inference:** تا زمانی که رأی QA76 کامل صادر نشده، **۷۶ مرکز candidate** و هر جملهٔ متکی به هندسهٔ ۴۰ raw مرکز (شامل مقایسه `I` و accuracy) باید از گزارش قطعی کنار گذاشته یا با عبارت «در انتظار ممیزی» مشخص شود. با `ACCEPT_QA76_CENTER_ONLY` فقط شواهد ۱۱ سنجهٔ مرکز برای هر ۷۶ اجرا ارتقا می‌یابند؛ آن رأی اصلاً به معنای تأیید خودکار `R_JS/R_KL/NLL/ties` نیست.

**P2 — تنها در صورت نیاز به ادعای جدید:** score/representation exact با *مجوز تازه*. صرف اینکه ۶۳۰ خانهٔ P1 `not_measured` هستند، الزامی برای ایجاد تمام مقادیر نیست؛ هر ادعا باید با حداقل داده و ارزش علمی توجیه شود. تضمین جلوگیری از Test leakage و حفظ query-start contract لازم است.

**Fail-closed:**
- **QA76 رد شود (اختلاف fingerprint/مرکز):** همهٔ **ادعاهای `center.*` مربوط به raw مشکل‌دار** و تمام مقایسه‌های وابسته به آن مرکز قرنطینه؛ اگر مشکل منشأ عمومی باشد، **کل بسته center-only و claims وابسته** تا reconciliation متوقف شود. اعداد historical مستقل از مرکز فقط با منشأ نسخهٔ خود باقی می‌مانند و به‌طور ضمنی validated-v2 نمی‌شوند. **هیچ ترمیم بی‌صدا یا جایگزینی مقدار تاریخی** مجاز نیست.
- **QA76 تنها `ACCEPT_WITH_LIMITATIONS` بماند:** ۳۶ مرکز با رأی محدود خود قابل ارجاع‌اند و ۴۰ مورد هنوز «candidate» هستند؛ P1 هنوز موقت و P2 به‌طور پیش‌فرض NO-GO.
- **QA76 قبول شود:** مسئول انتشار می‌تواند فقط center-only را برای ادعاهای مربوط ارتقا دهد، مشروط به ممیزی ledger ادعا؛ برای باقی ۴۴ شناسه، هر status باید **بدون ارتقای خودکار** به مرحلهٔ جداگانه بررسی برود.
- **P2 pilot شکست بخورد:** توقف، حفظ شواهد خام، عدم انتشار مقادیر جدید، ثبت اختلاف در issue و بازگشت برای تصمیم مدیر؛ نه تعمیم نتیجهٔ محدود به ۷۶ اجرا.

**تحویل P3 فقط چهار سند نسخه‌دار زیر `p3_publication_gates/` است**؛ بدون train، graph forward، replay، re-score، Test، تغییر تابع معیار/قرارداد، LaTeX، CI، PR/merge یا دستکاری artifactهای قدیمی. همهٔ خروجی‌ها همچنان **نقشهٔ بررسی گیت انتشار** هستند.

**RESEARCHER-03 — رأی نهایی: `READY_FOR_PUBLICATION_GATE_REVIEW`؛ توقف تا تصمیم مدیریت پژوهش.**
