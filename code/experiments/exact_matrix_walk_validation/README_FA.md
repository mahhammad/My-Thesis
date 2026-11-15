# گزارش فنی پژوهشی — E-CH4-08 | اعتبارسنجی بازگشت ماتریسی دقیق و کوتاه‌سازی گام‌ها

**شناسه:** `exact_matrix_walk_validation` | **وضعیت:** مطالعهٔ ثبت‌شده در فصل چهارم | **منشأ:** commit `e53b75cae25f`

## چکیده و دامنه

کنترل مستقل امید دقیق نسبت به MC و کران کوتاه‌سازی گام برای مدل ثابت H=12، روی دو سیاست shared ذخیره‌شده. این آزمون با تغییر سقف واقعی H متفاوت است.

فصل چهارم جایگاه گزارش نتایج علمی است؛ در این پیوست، قرارداد اجرایی، گواه‌های عددی، وضعیت فایل‌های ذخیره‌شده و مرز پذیرش علمی از هم تفکیک می‌شوند.

## گزارش علمی: روش، داده، سنجه‌ها، نتایج، نتیجه‌گیری

## E-CH4-08 — اعتبارسنجی محاسبه ماتریسی بازنمایی چندگامی

### سؤال و فرضیه
صحت امید دقیق در برابر گشت تصادفی مستقل، کران نظری حذف عمق ثابت H=12، و هزینه مشاهده‌شده گذر پیشرو بر مرجع کامل ۳۰۰تایی را می‌سنجیم. پیش‌ثبت در PROTOCOL_FA.md قبل از هر اجرا انجام شده است.

### داده، سیاست، شرایط
مولد استاندارد فصل چهارم: بذر 20261102، Train=300، Validation=90، Test تاریخی بارگذاری نشد. از کل ۳۰۰ نمونه در ترتیب reference_ids ذخیره‌شده در checkpoint Benchmark v1 استفاده شد؛ دو سیاست مشترکِ Z=0 و سیاست آموزش‌دیده ۵۰گامی (tau=.005) بدون هرگونه آموزش جدید بازسنجی شدند. beta=.4، lambda=.8، H=12، float64 و سه R/پنج جدایی/BA/NLL طبق فصل سوم و گزارشگر مشترک سنجیده شدند.

### اعتبارسنجی مستقل امید
در هر دو سیاست، پاسخ بازگشت مستقل در همه مبدأهای Train و Validation با موتور رسمی Benchmark v1 با خطای حداکثر مندرج در JSON تطبیق یافت. در شش مبدأ از پیش انتخاب‌شده، مونت‌کارلوی مستقل با بذرهای مسیر 20261121 و 20261122 و بودجه‌های 128/512/2048 برای ارزیابی تجربی به کار رفت. این‌ها دو بذر مولد مستقل نیستند.
- uniform: میانگین TV با M=128 برابر 0.03384 و با M=2048 برابر 0.01160 بود؛ این میانگین بر شش مبدأ از پیش مشخص و دو جریان MC است.
- trained_shared: میانگین TV با M=128 برابر 0.04106 و با M=2048 برابر 0.00990 بود؛ این میانگین بر شش مبدأ از پیش مشخص و دو جریان MC است.

### کران حذف عمق و کنترل مراکز
برای مدل اصلی H=12، نگه‌داشتن r گام نخست به‌معنای تغییر مدل به H=r نیست. در تمام ۳۰۰ بازنمایی Train، ۹۰ Validation و سه مرکز کلاسی، کران TV≤ε_r+1e-9 در تمام عمق‌های ازپیش‌ثبت‌شده و هر دو سیاست برقرار بود.
- uniform: r=3، کران ε=0.05911، میانگین متوازن TV آموزش=0.00558، Validation=0.00609؛ بیشینه TV مرکز=0.00644.
- trained_shared: r=3، کران ε=0.05911، میانگین متوازن TV آموزش=0.00552، Validation=0.00597؛ بیشینه TV مرکز=0.00575.

### پیچیدگی و هزینه
روی مرجع کامل شمار گره‌های نمونه–ویژگی 1366، تعداد یال‌های عمودی بالقوه 374024 و تعداد انتخاب‌های افقی بالقوه 6334 (شامل انتخاب مجاز همان ویژگی) است. هزینه نظری طرح تنک مربوط به ضرب بردار در عامل‌هاست، درحالی‌که پیاده‌سازی آزمایش از کرنل‌های متراکم Torch استفاده می‌کند. آرایه فرضی قدیمی n×d×n×n با float64 حدود 1.080 GB حافظه نیاز داشت؛ این آرایه در این اجرای Benchmark تخصیص داده نشده است. زمان‌های ثبت‌شده در tables/runtime.tex وابسته به ماشین و شامل آموزش نیستند.

### پوشش معیارهای عمومی
در results/metrics_full.json برای هر سیاست و هر r، Train/Validation دارای R_KL/R_JS/R_sqrt_JS، T/BT/TNL/BTNL/I، فاصله‌های زوجی مراکز، BA/Acc، ماتریس سردرگمی، حاشیه و انواع NLL است. دو snapshot خام imc-snapshot-v2 به همراه گزارش‌های مستقل ذخیره شده‌اند؛ فایل raw_step_distributions.npz تمام توزیع‌های گام‌به‌گام را نگه می‌دارد.

### محدودیت‌ها
این آزمایش تک‌بذر مولد داده است و تنها دو سیاست shared از پیش ثابت را بررسی می‌کند. بودجه‌های MC تو در تو هستند و برای تعمیم آماری مستقل نیستند. نتایج درباره کران محاسباتی‌اند، نه ادعای برتری دقت یا حل مسئله آموزش. حافظه RSS کل فرایند معادل حافظه یک نمونه/یک سیاست نیست. داده تاریخی ۷۲تایی منبع نتیجه رسمی نشده است. کامپایل کامل XeLaTeX و داوری مستقل هنوز لازم‌اند.

### راهنمای بازتولید و تحویل
با اجرای run.py --execute فقط امید دقیق/MC و گزارش عمومی روی Train/Validation تولید می‌شود؛ اجرا به هیچ وجه آموزش تازه یا بازخوانی Test ندارد. پوشه results شامل provenance.json، coverage.json، metrics_full.json، experiment_report.json، raw_step_distributions.npz و snapshotهای v2 است. تمام جدول‌ها از experiment_report.json تولید می‌شوند. وضعیت این شاخه فقط آماده بازبینی مستقل است و به chapter4 ادغام نشده است.


## پروتکل‌ها، بازتولید و یادداشت‌های مکمل

### منبع مستند تکمیلی: `PROTOCOL_FA.md`

## E-CH4-08 — پیش‌ثبت پروتکل اعتبارسنجی محاسبه ماتریسی چندگامی

**ثبت پیش از هر اجرای E08، ۱۰ نوامبر ۲۰۲۶؛ شاخه research/e08 از chapter4@b00fffdd41982e41c1b9d5a3e87ff344bbe8cbcd.** خروجی D08 تاریخیِ ۷۲نمونه‌ای مبنای گزارش رسمی این آزمایش نیست.

### پرسش و فرضیه‌های قابل ابطال

۱. آیا امید دقیقِ بازگشت ماتریسی با میانگین گشت مونت‌کارلو مستقل (با همان ماسک، حذف مبدأ، طول سقف‌دار و نرمال‌سازی وزن در هر گشت) سازگار است؟ H1: میانگین TV در بزرگ‌ترین بودجه MC کمتر از کوچک‌ترین بودجه در نمونه‌های ازپیش‌ثبت‌شده است. اختلاف بازگشت مستقل با موتور رسمی باید حداکثر 1e-9 باشد. کاهش یکایک خطاهای تصادفی تضمین نمی‌شود.

۲. برای مدل ثابت H=12، آیا تقریب rگامی که تنها αهای مدل اصلی را بازنرمال می‌کند برای هر نمونه Train/Validation و مراکز Train شرط TV≤ε_r+1e-9 را رعایت می‌کند؟ H2: همه مقادیر زیر این کرانند؛ تفاضل واقعی معمولاً کوچک‌تر از بدترین‌حالت است. این بحث با تعویض مدل اصلی به H=r متفاوت است.

۳. حذف گام انتهایی چه تأثیر قابل‌اندازه‌گیری بر زمان محاسبه دارد؟ H3: برای پیاده‌سازی و دستگاه مشخص، محاسبه r کوچک‌تر معمولاً سریع‌تر است، اما نسبت خطی/نتیجه مستقل از سخت‌افزار تضمین نمی‌شود.

### قرارداد ثابت قبل از اجرا

- مولد code/data_generator/config.json، بذر داده 20261102، مرجع *تمام* ۳۰۰ نمونه Train با **اندیس‌ها و ترتیب ثبت‌شده در checkpoint رسمی Benchmark** (به‌صورت یک جایگشت از کل ۳۰۰ اندیس؛ نه زیرمجموعه). سیاست یکنواخت نیز با همین ترتیب ارزیابی می‌شود؛ Validation ۹۰ نمونه. Test تاریخی به‌هیچ‌وجه تولید، بارگذاری یا برای انتخاب استفاده نمی‌شود. داده و هش config ثبت شوند.
- دو سیاست shared (هر دو شکل (d+1,d)): سیاست A = ماتریس Z صفر و سیاست B = checkpoint رسمی Benchmark v1 با n=300، objective=margin_JSD، tau=.005، steps=50، data_seed=20261102، فایل core_50_16runs.zip با کنترل SHA256 مانیفست. سیاست و ابرپارامتر جدید یاد گرفته نمی‌شود.
- beta=.4، lambda=.8، H=12، float64، gamma و ماسک مولد؛ ممنوعیت برگشت به مبدأ Train در تمام گام‌ها، حذف self عمودی، امتیاز شروع Validation بدون دسترسی به برچسب واقعی.
- عمق‌های تحلیل r∈{2,3,4,6,8,12}، همه با ضرایب مدل ثابت H=12 و ε_r=Σ_(t>r)α_t؛ نسخه r=12 پاسخ دقیق است.
- کنترل MC مستقل: اندیس مبدأ Train برابر [0,149,299] و Validation برابر [0,44,89]؛ بذرهای تصادف مسیر 20261121 و 20261122؛ تعداد گشت در هر نمونه M∈{128,512,2048}. تکرار MC به معنای بذر مستقل مولد داده نیست. وزن بردار مشاهده هر گشت با مجموع همان گشت نرمال شود.
- کنترل زمان: ۲۴ مبدأ Train با اندیس linspace(0,299,24)؛ r∈{2,4,12}، ۳ تکرار زمان‌سنجی و یک warm-up، Torch تک‌نخ روی CPU. زمان میانه گزارش شود؛ RSS فرایند و فرض آرایه متراکم قدیمی صریحاً جدا شوند.
- محاسبه مشترک: برای هر سیاست و r، مراکز Train از بازنمایی همان r و Validation در برابر مراکز Train ثابت ارزیابی می‌شوند؛ تمام سه R، T/BT/TNL/BTNL/I، هندسه زوجی مراکز، BA/Acc/Confusion، حاشیه/NLL از گزارشگر عمومی. TV فردی تا مرجع اصلی r=12 و TV مراکز به‌عنوان معیار تخصصی، بدون بهینه‌سازی بر اساس آنها.
- آرشیو رسمی دو سیاست با snapshot v2: model.json، distributions.npz و گزارش مشتق مستقل report.json/report.md، همه با منشأ و checksum. فایل خام تجربه و معیار تخصصی experiment_report.json، جدول‌های TeX متکی به اعداد ذخیره‌شده، شکل غیردوباره‌گو، coverage.json و provenance.json.
- جدول‌های چاپی فقط TV/ε/مدت و مشاهدات کلیدی، متن فصل سوم فقط ارجاع؛ از نتیجه این آزمایش، ادعای برتری طبقه‌بندی یا تعمیم چندبذری ساخته نشود.
- محدوده فایل‌های متن فقط زیربخش ۴.۸ در tex/chapter4.tex و پوشه جدید E08. chapter4 و فصل سوم دست‌نخورده؛ هیچ PR یا merge. کامپایل XeLaTeX و تأیید مستقل داور وضعیت جدا دارند.
- اجرای کد با پیش‌فرض dry-run و --execute صریح؛ پرهیز از هر آموزش/اجرای مجدد سنگین بی‌دلیل.

### مراجع خوانده‌شده

- tex/chapter3.tex: بخش امید دقیق و تقریب عمق محدود، روابط eq:ch3-capped-alpha، eq:ch3-matrix-forward-recursion و eq:ch3-matrix-truncation-bound.
- tex/chapter4.tex: بخش ۴.۲ قرارداد مشترک و متن اولیه ۴.۸.
- code/experiments/EXPERIMENT_REPORT_STANDARD_FA.md، COMMON_METRICS_FA.md، COMMON_METRICS_CONTRACT.json، REPORT_PRESENTATION_FA.md و benchmark_v1/PIPELINE_FA.md.
- آرشیو multistep_policy_target_match و bayes_center_inverse، موتور benchmark_v1/engine.py و گزارشگر گزارش استاندارد report_snapshot.py.

**شفاف‌سازی پیش از اجرا:** اندیس‌های MC، «موقعیت در ترتیب مرجع ثبت‌شده checkpoint» هستند، نه اندیس مرتب‌شده مولد؛ برچسب نمونه هرگز برای گزینش این موقعیت‌ها استفاده نمی‌شود.


### منبع مستند تکمیلی: `results/snapshots/uniform/report.md`

## گزارش استاندارد IMC

خانواده سیاست: shared
تابع هدف: uniform
پارامتر پهنای حاشیه برای گزارش: 0.005
داده Test تاریخی خوانده نشده است.

| معیار | Train | Validation |
|---|---:|---:|
| Balanced accuracy | 0.836462 | 0.967687 |
| Accuracy | 0.813333 | 0.955556 |
| I | 0.016883 | — |
| R_JS | 0.002014 | 0.001750 |
| R_sqrt_JS | 0.038460 | 0.038124 |
| R_KL | 0.008033 | 0.006994 |
| Sigmoid margin loss | 0.242934 | 0.143629 |
| JS-similarity NLL | 1.086818 | 1.084624 |
| Multiclass Softplus | 0.002077 | 0.000995 |

### محدودیت‌ها

وضعیت Validation: ok
خطای حاشیه در تساوی محافظه‌کارانه است؛ argmax اندیس اول را انتخاب می‌کند.
NLL نرمال KL تاریخی با صورت پایدار eta/epsilon فصل سوم یکسان نیست.


### منبع مستند تکمیلی: `results/snapshots/trained_shared/report.md`

## گزارش استاندارد IMC

خانواده سیاست: shared
تابع هدف: margin_JSD
پارامتر پهنای حاشیه برای گزارش: 0.005
داده Test تاریخی خوانده نشده است.

| معیار | Train | Validation |
|---|---:|---:|
| Balanced accuracy | 0.865195 | 0.972789 |
| Accuracy | 0.846667 | 0.955556 |
| I | 0.050590 | — |
| R_JS | 0.008351 | 0.006686 |
| R_sqrt_JS | 0.077148 | 0.074238 |
| R_KL | 0.032327 | 0.025742 |
| Sigmoid margin loss | 0.160721 | 0.054366 |
| JS-similarity NLL | 1.064109 | 1.060192 |
| Multiclass Softplus | 0.003211 | 0.000437 |

### محدودیت‌ها

وضعیت Validation: ok
خطای حاشیه در تساوی محافظه‌کارانه است؛ argmax اندیس اول را انتخاب می‌کند.
NLL نرمال KL تاریخی با صورت پایدار eta/epsilon فصل سوم یکسان نیست.


## راهنمای پیشین و نکات اجرایی

## E-CH4-08 — اعتبارسنجی مستقل محاسبه ماتریسی چندگامی

**وضعیت:** اجرا و آزمون‌های عددی موفق، ثبت نتایج خام روی `research/e08`؛ هنوز **ممیزی مستقل و ادغام انجام نشده‌اند**.

پیش‌ثبت علمی بدون بازنویسی در [PROTOCOL_FA.md](README_FA.md) و ژورنال پیش‌ثبت قرار دارد. تحلیل نهایی در [REPORT_FA.md](README_FA.md)، گردش تصمیم‌ها و اصلاح پسینی شمارش انتخاب‌های افقی در [Research Journal](../../../research_journal/2026-11-10-e08-execution-and-delivery.md) است.

### نتایج و منبع حقیقت

- [`results/experiment_report.json`](results/experiment_report.json): تمام نتایج تخصصی انتشار، برش در شش عمق، MC و زمان‌ها.
- [`results/metrics_full.json`](results/metrics_full.json): ۲۴ بلوک کامل معیار مشترک برای ۲ سیاست × ۶ عمق × Train/Validation.
- [`results/raw_step_distributions.npz`](results/raw_step_distributions.npz): داده خام گام‌های ۱ تا ۱۲ برای هر دو سیاست، Train و Validation.
- [`results/provenance.json`](results/provenance.json): SHA256 checkpoint/آرشیو، بذر، ترتیب ۳۰۰ مرجع و مسیرها.
- [`results/coverage.json`](results/coverage.json): همه بودجه‌ها، بذرها، مبدأها و پوشش متریک.
- `results/snapshots/{uniform,trained_shared}/`: در هر پوشه `model.json`، `distributions.npz` و `report.json/report.md` مطابق `imc-snapshot-v2`.
- `tables/` و `figures/`: خروجی‌های TeX ساخته‌شده **از فایل JSON ثبت‌شده**، استفاده‌شده در بخش ۴.۸ `tex/chapter4.tex`.

بذر داده **20261102**؛ MC با **20261121** و **20261122**؛ `beta=.4, lambda=.8, H=12`. فقط ۳۰۰ Train و ۹۰ Validation؛ Test تاریخچه خوانده نشده است. سیاست‌های یکنواخت و checkpoint مشترک ذخیره‌شده بدون آموزش تازه ارزیابی شده‌اند.

### بازتولید بدون اجرای مجدد

از ریشه مخزن:

```bash
python -m unittest discover -s code/experiments/exact_matrix_walk_validation -p 'test_*.py' -v
python code/experiments/exact_matrix_walk_validation/verify_results.py
python code/experiments/exact_matrix_walk_validation/run.py
```

دستور آخر به‌صورت پیش‌فرض **dry run** است. اجرای محاسبه جدید با `--execute` فقط در صورت نبود فایل نتایج موجود مجاز است و در صورت وجود نتایج متوقف می‌شود؛ اینجا نیازی به تکرار نیست.

CI اختصاصی [E08 matrix walk](https://github.com/mahhammad/My-Thesis/actions/runs/42804546810) با تست‌های سبک و بررسی read-only شواهد؛ در صورت موجود بودن `experiment_report.json` محاسبه جدید اجرا نمی‌کند. [اجرای موفق اصلی](https://github.com/mahhammad/My-Thesis/actions/runs/42804546810).

**یادداشت علمی:** شمار انتخاب‌های افقی در گراف باید خودحلقه مجاز ویژگی را در نظر بگیرد (`6334` روی مرجع کامل). گزارش اولیه `4968` فقط انتخاب‌های بین ویژگی‌های متفاوت را شمرد؛ این خطای *گزارش مشتق‌شده* بعداً بدون محاسبه مجدد تصحیح شد. مسیر و تصمیم در Research Journal محفوظ‌اند.


## راهنمای کامل فایل‌های موجود

کدهای علمی، فایل‌های raw/نتایج و خروجی‌های TeX/SVG همگی در محل خود باقی می‌مانند. فایل‌های Markdown قبلی در همین سند ادغام شده‌اند.

| پرونده (نسبت به ریشهٔ این مطالعه) | نقش |

|---|---|

| [`figures/bound_and_max_tv.tex`](figures/bound_and_max_tv.tex) | جدول/منبع TeX قابل درج در فصل |

| [`presentation.py`](presentation.py) | ماژول/ابزار پردازش داده و شواهد |

| [`results/coverage.json`](results/coverage.json) | پوشش معیارها و وضعیت اندازه‌گیری |

| [`results/experiment_report.json`](results/experiment_report.json) | نتایج عددی و سنجه‌های طبقه‌بندی/هندسه |

| [`results/metrics_full.json`](results/metrics_full.json) | نتایج عددی و سنجه‌های طبقه‌بندی/هندسه |

| [`results/provenance.json`](results/provenance.json) | مشخصات نسخه و هش منشأ/هویت داده |

| [`results/raw_step_distributions.npz`](results/raw_step_distributions.npz) | آرایه‌های محاسباتی و توزیع‌های snapshot ذخیره‌شده |

| [`results/snapshots/trained_shared/distributions.npz`](results/snapshots/trained_shared/distributions.npz) | آرایه‌های محاسباتی و توزیع‌های snapshot ذخیره‌شده |

| [`results/snapshots/trained_shared/model.json`](results/snapshots/trained_shared/model.json) | داده ساختاریافته، سیاست یا پیکربندی محاسباتی |

| [`results/snapshots/trained_shared/report.json`](results/snapshots/trained_shared/report.json) | نتایج عددی و سنجه‌های طبقه‌بندی/هندسه |

| [`results/snapshots/uniform/distributions.npz`](results/snapshots/uniform/distributions.npz) | آرایه‌های محاسباتی و توزیع‌های snapshot ذخیره‌شده |

| [`results/snapshots/uniform/model.json`](results/snapshots/uniform/model.json) | داده ساختاریافته، سیاست یا پیکربندی محاسباتی |

| [`results/snapshots/uniform/report.json`](results/snapshots/uniform/report.json) | نتایج عددی و سنجه‌های طبقه‌بندی/هندسه |

| [`run.py`](run.py) | پیاده‌سازی محاسبه، گواه یا حلگر آزمایش |

| [`tables/monte_carlo.tex`](tables/monte_carlo.tex) | جدول/منبع TeX قابل درج در فصل |

| [`tables/runtime.tex`](tables/runtime.tex) | جدول/منبع TeX قابل درج در فصل |

| [`tables/truncation.tex`](tables/truncation.tex) | جدول/منبع TeX قابل درج در فصل |

| [`test_e08_numerical.py`](test_e08_numerical.py) | آزمون خودکار برای کنترل عددی، قرارداد داده یا بازتولید |

| [`test_e08_presentation.py`](test_e08_presentation.py) | آزمون خودکار برای کنترل عددی، قرارداد داده یا بازتولید |

| [`verify_results.py`](verify_results.py) | کد ممیزی، بازپخش و تحلیل دادهٔ ذخیره‌شده |

| [`walk_analysis.py`](walk_analysis.py) | کد ممیزی، بازپخش و تحلیل دادهٔ ذخیره‌شده |

| `README_FA.md` | گزارش فنی تجمیعی، راهنمای فایل‌ها و شواهد |

## بازیابی سوابق Markdown حذف‌شده

نسخه‌های قبلی گزارش/پروتکل و شواهد متنی، با Git blob اصلی در [کامیت پیش از تجمیع](https://github.com/mahhammad/My-Thesis/tree/e53b75cae25ffba0d06d7b8a15f6cf322cb4698f/code/experiments/exact_matrix_walk_validation/) باقی می‌مانند. ارجاعات موجود در کامنت‌های تاریخی Issueها به همین نسخه‌ها تعلق دارند.
