# E-CH4-13 — گزارش پایلوت مقایسه کنترل‌شده دو تابع زیان IMC
**RESEARCHER-06 | ۱۴ نوامبر ۲۰۲۶ | وضعیت: PILOT_READY_FOR_QA / PROVISIONAL — NOT INDEPENDENTLY ACCEPTED**

**مرجع مأموریت:** [Issue #78](https://github.com/mahhammad/My-Thesis/issues/78)؛ رأی `ACCEPT_PREFLIGHT_DESIGN_WITH_CONDITIONS` در [کامنت مدیر](https://github.com/mahhammad/My-Thesis/issues/78#issuecomment-6880231186). **مرجع خام:** [results/pilot_seed20261102_shared_two_fits/](results/pilot_seed20261102_shared_two_fits/). **PR:** [#79](https://github.com/mahhammad/My-Thesis/pull/79)، Draft؛ ممنوعیت مرج برقرار است.

## ۱. پرسش و حدود ادعا
آیا تابع سیگموئیدی حاشیه JS و Softplus چندرقیبی JS تحت داده، خانواده سیاست، مقداردهی اولیه، تعداد به‌روزرسانی و τ ثابت، مسیر یادگیری و BA متفاوتی می‌سازند؟ این پایلوت برای صحت pipeline و مشاهده رفتار طراحی شده؛ **یک seed و ۱۰ گام هیچ برتری آماری یا انتخاب نهایی تابع زیان را اثبات نمی‌کند**. هر اختلاف می‌تواند به مقداردهی اولیه، مقیاس گرادیان، داده، checkpoint یا محدودیت بهینه‌سازی وابسته باشد.

## ۲. روش محاسباتی و قرارداد
- مولد مصنوعی استاندارد سه‌کلاسه/پنج‌ویژگی، فقط Train/Validation؛ seed داده `20261102`، زیرمجموعه طبقه‌متناسب `n=50` با reference seed `20261107` و reference_ids قفل‌شده در manifest؛ Validation=۹۰. هیچ Test تولید/ارزیابی نشده است.
- سیاست `shared` به شکل `(d+1,d)`، شروع صفر جفت‌شده. Adam، نرخ یادگیری `0.12`، دقیقاً **۱۰ update** برای هر objective، `float64`، گشت امید دقیق با `β=.4`، `λ=.8`، `H=12`، ماسک موجود، منع بازگشت به نمونه مبدأ.
- امتیاز نامزد از شباهت نرمال‌شده JS به مرکز Train و برچسب نامزد مستقل از حقیقت Validation استفاده می‌کند. تعریف معیارهای عمومی به فصل سوم `tex/chapter3.tex` و `COMMON_METRICS_CONTRACT.json` ارجاع دارد. مراکز Validation بازبرازش نشده‌اند.
- تابع اول `ℓτ(m)=σ(−m/τ)` با `τ=.005` و کاهش متوازن روی کلاس‌ها؛ تابع دوم `τ·logsumexp_c((s_c−s_y)/τ)` با همان `τ=.005` و کاهش متوازن. پیاده‌سازی آموزشی از `replay_v1.objective_loss` و `train_missing_objectives.train_loss` canonical reuse می‌کند. **مقیاس عددی و گرادیان دو objective برابر نیست**؛ برابر بودن optimizer و steps مساوی بودن اثر آموزشی نیست.
- snapshotها طبق `imc-snapshot-v2` (دو فایل `model.json` و `distributions.npz`)، همراه `metrics_full.json`، `trace.json` و SHA256 هر فایل ذخیره شدند. گزارشگر `report_snapshot.py` raw را مستقلاً با Train/Validation و fingerprintها بازخوانی می‌کند.

## ۳. نتیجه مشاهده‌شده (پایلوت؛ بدون ادعای برتری)
اعداد جدول از `results/pilot_seed20261102_shared_two_fits/manifest.json` و `metrics_full.json` بازخوانی شده‌اند. BA به درصد، هندسه و تراکم بدون بعد:

| هدف آموزش | Train BA (%) | Validation BA (%) | I ↑ | R_JS ↓ | R_KL ↓ | R_sqrt_JS ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Sigmoid-JS τ=.005 | 78.7607 | **87.7551** | .063964 | .008345 | .032668 | .070982 |
| Softplus-JS τ=.005 | **83.9915** | 82.6400 | .021570 | **.002669** | **.010484** | **.040228** |

اختلاف مشاهده‌شده Validation BA برابر **۵٫۱۱۵ واحد درصد به سود سیگموئید** است؛ در عوض Softplus در Train BA حدود **۵٫۲۳ واحد درصد بالاتر** دارد و سه تراکم R پایین‌ترند. در این نمونه کاهش پراکندگی `R_JS` و افزایش دقت آموزش، همراه افزایش دقت Validation نبوده‌اند. سیگموئید `I` بالاتر و پراکندگی درون‌کلاسی بیشتر دارد. این شواهد محدود، مستقل از بافت پایلوت نتیجه قطعی درباره رابطه `R/I/BA` نیستند.

**حاشیه تصمیم Validation:** میانه حاشیه برای سیگموئید `0.029507` و برای Softplus `0.006437` است. هیچ تساوی گزارش نشده. ماتریس سردرگمی با سطر حقیقت و ستون پیش‌بینی:

- Sigmoid: `[[13,0,0],[5,38,6],[0,4,24]]`؛ Recall کلاس‌های ۰/۱/۲ به‌ترتیب `1.0000, .7755, .8571`.
- Softplus: `[[12,1,0],[6,36,7],[0,5,23]]`؛ Recall `.9231, .7347, .8214`.

مقادیر NLL، Softplus، sigmoid، تمام پنج جدایی `T/BT/TNL/BTNL/I`، ماتریس فاصله مراکز، تمام Rها و توزیع حاشیه در `metrics_full.json` هر objective موجود است. فایل `coverage.json` برای ۴۳ شناسه split/metric در هر objective مقدار `measured` دارد؛ Test به‌صورت `not_applicable` و سنجش پیک حافظه `not_measured` دلیل‌دار ثبت شده‌اند.

## ۴. زیان، گرادیان و محدودیت مقیاس
| هدف | زیان ابتدا | زیان پایان | norm2 گرادیان ابتدا | norm2 گرادیان پایان |
|---|---:|---:|---:|---:|
| Sigmoid-JS | .3482500 | .2201254 | .1063069 | .0181370 |
| Softplus-JS | .00322562 | .00277090 | .000658662 | .000113791 |

هر دو مقدار loss مخصوص خودشان کاهش یافته‌اند، ولی این lossها کمیت هم‌مقیاس نیستند. norm گرادیان آغاز سیگموئید تقریباً ۱۶۱ برابر Softplus است؛ **مقایسه عدد خام loss دو خانواده بی‌معناست**. تمام ۱۱ نقطه از آغاز `step=0` تا `step=10` در `trace.json` هر هدف ثبت شده؛ checkpoint انتخاب‌شده از پیش endpoint نهایی بوده، نه بیشینه Validation.

## ۵. بازتولید، PASS/FAIL/NOT_RUN و خطاهای واقعی
- گیت ۲۲ آزمون پیش از اجرا در [CI #43614497613](https://github.com/mahhammad/My-Thesis/actions/runs/43614497613) **PASS**: ۸ اختصاصی + ۴ قرارداد معیار + ۷ snapshot + ۳ roundtrip. `run_experiment.py` پیش‌فرض DRY_RUN است و بدون محیط `E_CH4_13_PREFLIGHT_PASS=true` و `--execute` آموزش نمی‌دهد.
- اجرای اصلیِ دو fit در همان CI انجام شد، اما **job به علت خطای کدنویسی پس از fit در ساخت coverage.json FAIL شد** (تبدیل dictionary چندک‌های حاشیه به float). این شکست پنهان نشده است: [artifact raw اولیه](https://github.com/mahhammad/My-Thesis/actions/runs/43614497613/artifacts/14121258818) شامل ۸ فایل immutable حفظ شد؛ SHA256 ZIP در log: `d243b375f4b0034d31069760dbc0606dfbd93ea91ca28965a732d2c6708ee0d1`.
- **صفر آموزش اضافی:** فرایند `finalize_existing.py` فقط همان فایل‌ها را از artifact اولیه بازخوانی کرده و مدل/داده، fingerprint، trace، hash و گزارشگر را کنترل نموده و سپس manifest/coverage/CSV/SVG را محاسبه کرده است. [CI بازیابی موفق #43615046534](https://github.com/mahhammad/My-Thesis/actions/runs/43615046534) **PASS**؛ git commit بازیابی `58d1430519c31602269fc8eaaf7f8ea15ed5da98`. عددهای جدید از **آموزش جدید** نیستند.
- تلاش نخست بازیابی [#43614895487](https://github.com/mahhammad/My-Thesis/actions/runs/43614895487) با تشخیص کلی fingerprint mismatch متوقف شد؛ پس از افزودن گزارش تفکیکی هریک از چهار fingerprint، اجرای بعدی همه کنترل‌ها را PASS کرد. علت قطعی تفاوت دو تلاش مشخص نشده؛ این **ریسک بازبینی مستقل منشأ** است، نه مجوز چشم‌پوشی از کنترل‌ها. فایل خام اولیه دست‌نخورده مانده است.
- تست‌های heavy/full300/multiseed/Test و QA علمی مستقل و XeLaTeX/PDF: **NOT_RUN**. هیچ به‌روزرسانی `tex/` یا فایل تاریخچه‌ای انجام نشده. ادعای پذیرش غیرمرکزمحور QA76 داده نشده است.

## ۶. مسیر فایل‌ها و گیت جانشینی
- `results/pilot_seed20261102_shared_two_fits/manifest.json` و `coverage.json`، `metrics_summary.csv`، `validation_ba.svg`
- `results/pilot_seed20261102_shared_two_fits/<objective>/model.json`، `distributions.npz`، `trace.json` و `metrics_full.json`، همگی با hashes در manifest
- کد اجرا: `run_experiment.py`، کد بدون آموزش بازیابی `finalize_existing.py`، آزمون سبک `test_e_ch4_13_preflight.py`، گیت GitHub Actions `.github/workflows/e-ch4-13-*.yml`.
- گزارش علمی تنها نتیجه مقدماتی این پایلوت را بیان می‌کند؛ مانع انجام/ثبت خطاهای فرایندی نیست. پس از گیت مدیر، تفسیر در بخش ۴.۱۳ و شکل/جدول پایان‌نامه فقط با قبول QA مستقل جایز است.

**گیت مورد درخواست:** `PILOT_READY_FOR_QA`، نه ACCEPTED و نه مجوز اجرای اصلی. مدیر باید raw و manifest، تکرار بی‌آموزش fingerprint و پوشش معیارها، مقیاس گرادیان، ارتباط با فصل سوم و کد را مستقلاً ممیزی کند. **۱۸/۲۴ fit و Test/مرج ممنوع‌اند.**
