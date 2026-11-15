# E-CH4-13 — مقایسه کنترل‌شده دو تابع زیان IMC در ۳۰۰ نمونه

**وضعیت علمی:** `PHASE2_READY_FOR_INDEPENDENT_QA` / نتایج تجربی مقدماتی، بدون ادعای پذیرش مستقل.  
**مأموریت:** [Issue #81](https://github.com/mahhammad/My-Thesis/issues/81)؛ **اجرای اصلی:** [Actions #43670524935](https://github.com/mahhammad/My-Thesis/actions/runs/43670524935) (SUCCESS، یک‌بار).  
**SHA موتور آموزش قفل‌شده:** `cdb45644c245e04a276eb2a419e4d64f71c6c18a`.  
**منبع خام نهایی:** [`results/phase2_300_25/`](results/phase2_300_25/) و [GitHub artifact #14144445918](https://github.com/mahhammad/My-Thesis/actions/runs/43670524935/artifacts/14144445918).  
**عدم دسترسی به Test:** تأییدشده از قرارداد اجرا و provenance؛ متن نهایی پایان‌نامه (`tex/`) دست‌نخورده است و PR #79 همچنان Draft است.

## ۱. پرسش، فرضیه و محدودیت

در IMC آیا تابع زیان `Sigmoid-JS margin` نسبت به `Softplus-JS` دقت متوازن Validation بهتری می‌سازد و آیا تراکم کلاس‌ها (`R`)، جدایی مراکز (`I`) و حاشیه تصمیم، تغییر تعمیم را توضیح می‌دهند؟ مقایسه **جفتی** و از پیش ثبت‌شده است؛ تغییر علامت اختلاف بین مولدها مجاز است و نباید برای انتخاب برندهٔ جهانی از همان سه مولد استفاده شود. آزمایش برای آزمون حساسیت به seed و policy است، نه اثبات برتری، همگرایی جهانی یا significance تعمیم‌پذیر.

## ۲. قرارداد ثابت و تعداد اجرا

سه بذر مولد مصنوعی `20261102/03/04` × دو خانواده سیاست `shared/class` × دو تابع زیان = **۱۲ fit واقعی** و شش جفت. هر fit: `nTrain=300`, `nValidation=90`, `K=3`, `d=5`, `beta=0.4`, `lambda=0.8`, `H=12`, موتور exact float64 با گشت بدون بازگشت، Adam (`lr=0.12`)، logits اولیه صفر، **دقیقاً ۲۵ به‌روزرسانی**؛ `tau_width=0.005`. توقف زودهنگام، جست‌وجوی ابرپارامتر و انتخاب بهترین epoch بر اساس Validation ممنوع؛ checkpoint نهایی همان گام ۲۵ است.

فایل‌های frozen Train/Validation هر بذر با checksum منجمدند. تعریف score برابر `1-JSD/ln(2)` است. Sigmoid زیان سیگمویدِ حاشیه درست در مقابل نزدیک‌ترین رقیب را به‌طور متوازن کلاس‌ها کمینه می‌کند؛ Softplus نسخه `tau*logsumexp` همه فرض‌های رقیب را. هر دو همان engine و قرارداد گزارش معیارهای مشترک را استفاده کرده‌اند.

## ۳. نتیجهٔ جفتی Validation BA

علامت `ΔBA = BA(Sigmoid) − BA(Softplus)`. ستون‌ها درصد و اختلاف‌ها **واحد درصد** هستند. `flip` تعداد تغییر پیش‌بینی میان دو مدل در همان Validation است.

| بذر | policy | Sigmoid BA | Softplus BA | ΔBA (pp) | flip |
|---|---|---:|---:|---:|---:|
| 20261102 | shared | 97.9592 | 96.7687 | +1.1905 | 1 |
| 20261102 | class | 97.9592 | 97.9592 | 0.0000 | 4 |
| 20261103 | shared | 87.7391 | 86.4571 | +1.2821 | 1 |
| 20261103 | class | 81.2068 | 85.8873 | −4.6805 | 8 |
| 20261104 | shared | 83.8914 | 86.0419 | −2.1505 | 4 |
| 20261104 | class | 84.1536 | 86.0419 | −1.8883 | 4 |

**میانگین اختلاف جفتی** در خانواده shared: `+0.1073 pp` با `sample SD=1.9559 pp` و علامت `+,+,−`؛ در خانواده class: `−2.1896 pp` با `sample SD=2.3548 pp` و علامت `0,−,−`. تنها **سه بذر مستقل در هر خانواده** وجود دارد؛ انحراف معیار توصیفی است و داده‌های درون یک مولد یا گام‌های optimizer نمونه‌های مستقل محسوب نمی‌شوند.

## ۴. تراکم، جدایی مراکز و حاشیه

تمام شش جفت **Softplus دارای `R_JS` و `R_KL` کمتر در Train** و **Sigmoid دارای `I` و میانه حاشیه Validation بالاتر** بودند. اما جهت BA به مولد و خانواده وابسته است.

| بذر | policy | I Sig | I Soft | R_JS Sig | R_JS Soft | median margin Val Sig | median margin Val Soft |
|---|---|---:|---:|---:|---:|---:|---:|
| 20261102 | shared | 0.04800 | 0.01867 | 0.00778 | 0.00252 | 0.03431 | 0.01234 |
| 20261102 | class | 0.03802 | 0.01912 | 0.00858 | 0.00266 | 0.03377 | 0.01279 |
| 20261103 | shared | 0.06521 | 0.01969 | 0.00703 | 0.00224 | 0.03881 | 0.01265 |
| 20261103 | class | 0.13554 | 0.02036 | 0.01391 | 0.00232 | 0.06191 | 0.01380 |
| 20261104 | shared | 0.04676 | 0.01827 | 0.00796 | 0.00265 | 0.03606 | 0.01227 |
| 20261104 | class | 0.04764 | 0.01741 | 0.00615 | 0.00259 | 0.03581 | 0.01270 |

نمونهٔ خلاف تفسیر یک‌بعدی: در `seed=20261103, class` با اینکه Sigmoid جدایی مراکز و میانه حاشیه بالاتری دارد، **Validation BA آن ۴٫۶۸ واحد درصد کمتر از Softplus** است. برعکس در `seed=20261102, shared`، Sigmoid برتری کوچک BA دارد. بنابراین نه `R↓` و نه `I↑` یا margin↑ به‌تنهایی تضمین‌کننده بهبود BA نیستند. اثر توزیع نمونه‌ها و هندسه تصمیم باید همراه confusions، recall کلاس‌ها، مرز تصمیم و مقدار واقعی margin مطالعه شود. حتی `median margin` بالاتر الزاماً تعداد خطاهای کمتر را تضمین نمی‌کند.

مقیاس گرادیان دو loss متفاوت است؛ برابر بودن ۲۵ گام Adam و learning rate به معنای برابر بودن سرعت همگرایی، مقدار gradient یا ظرفیت مدل نیست. ادعای برتری عمومی یا استنباط p-value/CI قوی از سه seed صورت نمی‌گیرد.

## ۵. معیارها و ردپای بازتولید

هر یک از ۱۲ پرونده دارای `model.json`، `distributions.npz`، `trace.json`، `metrics_full.json`، `coverage.json`، `run_evidence.json` است. گزارش‌های `metrics_full.json` شامل گروه density (`R_KL/R_JS/R_sqrt_JS`)، هندسه (`I/T/BT/TNL/BTNL`)، BA/Acc، confusion و per-class recall، پیش‌بینی‌ها و خطاها، margin و quantileهای آن، معیارهای جانشین زیان، و مقادیر per-sample هستند؛ `NLL`/surrogateهای مرتبط و حدود تعریف‌پذیری نیز گزارش شده‌اند.

- `manifest.json` با **۱۲ رکورد و ۶۰ هش SHA256** از فایل‌های مرکزی هر fit ذخیره شده و تمام ۶۰ تطبیق بایتی در بررسی مستقل از ZIP PASS شد. checksumِ داده‌ها و provenance هم تطبیق داشتند.
- ۱۲ نشانگر مصرف بودجه، ۱۲ ledger checkpoint، ۱۲ success checkpoint، **۳۱۲ checkpoint گام‌به‌گام** و دقیقاً ۲۶ سطر trace (گام‌های ۰ تا ۲۵) در هر fit ثبت شده‌اند؛ هیچ فایل `failures/` وجود ندارد.
- بازمحاسبه مستقل `JSD→score` و `argmax` بر اساس `distributions.npz`، بازشماری confusion و BA از `per_sample` و تطبیق با manifest و `paired_summary.json` در هر ۱۲ fit بدون اختلاف انجام شد.
- **نکته پوشش معیار:** `coverage.json` دو مسیر `train.per_class_recall` و `validation.per_class_recall` را به علت نام کلید `not_measured` می‌نامد، ولی مقادیر واقعی per-class recall در `metrics_full.json/per_class[].accuracy` و confusion موجودند. این اختلاف نام‌گذاری/coverage در ممیزی مستقل باید بررسی شود. داده خام به همین شکل اصلی حفظ شده و پوشش به‌صورت ساختگی «PASS کامل» اعلام نشده است.
- سیستم عددی `Python 3.11.18`, `NumPy 2.4.6`, `torch CPU 2.15.0`, `OPENBLAS_CORETYPE=generic` و تک‌ریسمانی در `provenance.json` ثبت شده؛ گیت SHA و تأیید رسمی مدیر در فایل مجوز ثبت و token محرمانه به raw وارد نشده است.

### منشأ غیرقابل تغییر

- موفقیت آموزش، reducer و آپلود: [Run #43670524935](https://github.com/mahhammad/My-Thesis/actions/runs/43670524935).
- نام artifact اصلی: `e-ch4-13-phase2-partial-or-complete-43670524935-1`؛ ID `14144445918`؛ SHA256 فایل ZIP برابر `26e7a94e36edeae2a3153b76b40cf282c30b550a3fac359558a20d11e72c3e3e`.
- بایگانی در شاخه آزمایش صرفاً از **بایت‌های خام artifact** و به کمک [`archive_phase2_existing.py`](archive_phase2_existing.py) انجام شد؛ **هیچ آموزش جدید** برای گزارش یا بازیابی اجرا نشد.
- [CSV دوازده اجرا](tables/phase2_runs_12.csv)، [CSV شش جفت](tables/phase2_pairs_6.csv) و [نمودار ΔBA](figures/phase2_paired_delta_validation_ba.svg) تنها از manifest/pair raw تولید شده‌اند. گزارشهای اضافی و ترسیم تک‌خانواده‌ای می‌توانند بعد از QA بدون تغییر raw ساخته شوند.

## ۶. جمع‌بندی و گیت انتشار

مقایسهٔ دوازده‌تایی با حفظ پروتکل پایان یافته و شواهد نشان می‌دهند ارتباط `R`، `I`، حاشیه و BA حساس به مولد و خانواده سیاست است. این یافته برای فصل چهارم مهم است ولی با سه مولد مصنوعی و بدون داده Test **نتیجه نهایی درباره تعمیم** یا انتخاب تابع زیان برتر نیست. درخواست از ممیز مستقل در [Issue #80](https://github.com/mahhammad/My-Thesis/issues/80): بازبینی شواهد بایتی، اندازه‌گیری metric coverage و هماهنگی برچسب‌های `per_class_recall`، صحت losses و engine، عدم نشت Validation/Test، اعداد raw-derived و اعتبار استنباط. فقط پس از رأی ممیزی و دستور مدیر، نگارش `tex/chapter4.tex` مجاز است.

**وضعیت:** `PHASE2_READY_FOR_INDEPENDENT_QA`؛ نه `ACCEPT_SCIENTIFIC`. PR #79 نباید ادغام شود و Test همچنان `NOT_RUN` است.
