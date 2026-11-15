# E-CH4-13 — مقایسه کنترل‌شده تابع زیان IMC

**مالک:** RESEARCHER-06 · **مدیر:** RESEARCH-MANAGER-04 · [Issue #78](https://github.com/mahhammad/My-Thesis/issues/78) · [Draft PR #79](https://github.com/mahhammad/My-Thesis/pull/79)

## وضعیت جاری (۱۵ نوامبر ۲۰۲۶): PHASE2_READY_FOR_INDEPENDENT_QA — اجرای ۱۲ fit کامل، پذیرش علمی در انتظار

**اجرای موفق:** [Actions #43670524935](https://github.com/mahhammad/My-Thesis/actions/runs/43670524935)؛ **SHA آموزش قفل‌شده:** `cdb45644c245e04a276eb2a419e4d64f71c6c18a`؛ **بایگانی ۱۲ fit:** [`results/phase2_300_25/`](results/phase2_300_25/) با checksum، ۱۲ snapshot، traceهای ۰..۲۵، ۳۱۲ checkpoint optimizer، داده‌های Train/Validation و بدون Test. بازیابی و بایگانی فقط از [artifact اصلی #14144445918](https://github.com/mahhammad/My-Thesis/actions/runs/43670524935/artifacts/14144445918) و **بدون آموزش مجدد** انجام شد.

- [`REPORT_PHASE2_FA.md`](REPORT_PHASE2_FA.md): گزارش علمی کامل، طراحی قفل‌شده، مقایسه ۶ جفت، **هر سه R: `R_KL`، `R_JS`، `R_sqrt_JS` روی Train و Validation**، `I`، margin، BA، تحلیل روند آموزش و محدودیت‌های استنباط؛ توضیح `ExactMaskedWalk` در افق `H=12` و **MC paths = N/A**.
- [`tables/phase2_runs_12.csv`](tables/phase2_runs_12.csv) و [`tables/phase2_pairs_6.csv`](tables/phase2_pairs_6.csv): دو جدول مشتق‌شده از داده واقعی.
- [`figures/`](figures/): **۱۱ نمودار تحلیلی جدید**: سه نسخه `R` جداگانه برای Train و Validation (۶ نمودار)، اختلاف BA، رابطه `ΔI` با `ΔBA`، روند نسبی loss و gradient و روند Train BA؛ افزون بر دو تصویر تاریخی/اولیه.
- [`phase2_preregistration.md`](phase2_preregistration.md) و [`phase2_protocol.py`](phase2_protocol.py): طراحی و قرارداد فریز‌شده؛ [`PHASE2_RAW_ARCHIVE_CHECKPOINT.json`](PHASE2_RAW_ARCHIVE_CHECKPOINT.json): ثبت مبدأ بایگانی اصلی.
- **اصلاحات رأی ممیز [#80](https://github.com/mahhammad/My-Thesis/issues/80#issuecomment-6889701832):** پوشش اصلاحی نسخه‌دار در [derived/phase2_qa_fixes_v2/coverage/](derived/phase2_qa_fixes_v2/coverage/)، کد تولید [phase2_derived_reproduce_v2.py](phase2_derived_reproduce_v2.py)، **۱۱ نمودار قابل بازتولید نسخه دوم** در [figures/phase2_repro_v2/](figures/phase2_repro_v2/)، دو جدول CSV و [مانیفست SHA نمودارها](derived/phase2_qa_fixes_v2/plot_figure_manifest.json) ثبت شده‌اند. همه شکل‌های قبلی و raw اصلی حفظ شده‌اند. نتیجه همچنان **منتظر ممیزی مستقل مجدد** است.
- **QA مستقل علمی هنوز باقی است:** [Issue #80](https://github.com/mahhammad/My-Thesis/issues/80). وضعیت `coverage.json` برای `per_class_recall` باید با مقادیر واقعی `metrics_full.json/per_class` تطبیق داده شود. **Test=NOT_RUN؛ PR #79 Draft؛ LaTeX و خام تاریخی بدون تغییر.**
- **اجرا/تکرار ممنوع:** بودجه ۱۲ fit مصرف شده است؛ فایل `phase2_execution_gate.json` متعلق به طرح قفل‌شده تاریخی و تغییرناپذیر است و برای شروع آموزش دیگری نباید استفاده شود.

## وضعیت تاریخی پایلوت: PILOT_READY_FOR_QA / UNVERIFIED SCIENTIFIC CLAIMS
طرح پیش‌پرواز [پذیرفته‌شده با شروط](https://github.com/mahhammad/My-Thesis/issues/78#issuecomment-6880231186) و **دقیقاً دو آموزش مجاز، هرکدام ۱۰ update** روی seed=20261102 و shared انجام شد. گزارشگر مرحله نهایی در اجرای اولیه خطای پوشش متریک داشت؛ فایل‌های هر دو run به‌صورت artifact محفوظ ماندند و بدون اجرای optimizer مجدد [بازیابی و راستی‌آزمایی شدند](https://github.com/mahhammad/My-Thesis/actions/runs/43615046534).

**در زمان نگارش این بخش تاریخی، مطالعه کامل چندبذری، full300، Test و merge هنوز مجاز نبود.** نتیجه مقدماتی فقط مربوط به یک مولد و بودجه کم است؛ QA مستقل علمی انجام نشده است.

## مسیر مطالعه و تحویل
- [RESEARCHER-06_READINESS_REPORT.md](RESEARCHER-06_READINESS_REPORT.md): گزارش پیش‌پرواز اولیه (checkpoint تاریخی قبل از پایلوت)
- [preregistration.md](preregistration.md): طرح و فرضیه کنترل‌شده
- [REPORT_FA.md](REPORT_FA.md): گزارش کامل پایلوت، نتایج، شواهد منفی و حدود استنباط
- [inventory_existing.csv](inventory_existing.csv): موجودی مقایسه‌های سابق
- [PILOT_EXECUTION_AUTHORIZATION_FA.md](PILOT_EXECUTION_AUTHORIZATION_FA.md): گیت و بودجه مصوب
- [RECOVERY_CHECKPOINT_FA.md](RECOVERY_CHECKPOINT_FA.md): گزارش شکست واقعی پس از fit و بازیابی بی‌آموزش
- [results/pilot_seed20261102_shared_two_fits/](results/pilot_seed20261102_shared_two_fits/): raw، fingerprint، hash، trace، coverage، کامل‌ترین metric report، CSV و SVG
- [tables/pilot_summary.tex](tables/pilot_summary.tex) و [figures/validation_ba.svg](figures/validation_ba.svg): منابع مشتق‌شده از همان raw

## اجرای مجدد بدون آموزش برای QA
```bash
python -m unittest discover -v -s code/experiments/ch4_loss_controlled_comparison -p 'test_*.py'
python -m unittest discover -v -s code/experiments/benchmark_v1 -p 'test_common_metrics_contract.py'
python -m unittest discover -v -s code/experiments/benchmark_v1 -p 'test_snapshot_pipeline.py'
python -m unittest discover -v -s code/experiments/benchmark_v1 -p 'test_snapshot_roundtrip.py'
```
`finalize_existing.py` تنها raw موجود را بازخوانی و خروجی دیجیتال را از آن بازسازی می‌کند؛ در آن optimizer وجود ندارد. `run_experiment.py` به‌طور پیش‌فرض dry-run و در حالت --execute fail-closed است؛ با پایان بودجه دو fit این مرحله مجوز تکرار آن وجود ندارد.

**درخواست گیت:** `PILOT_READY_FOR_QA`؛ هیچ پذیرش نتیجه علمی یا مجوز مرحله اصلی پیش از رأی ممیزی مستقل.

## بازتولید عددی با محیط ثابت
در ممیزی بعدی مشخص شد که توابع تولید داده NumPy می‌توانند با dispatch مختلف OpenBLAS خروجی بایت‌به‌بایت متفاوت تولید کنند، حتی وقتی seed/config/برچسب‌ها یکسان‌اند. [تشخیص CI](https://github.com/mahhammad/My-Thesis/actions/runs/43615893696) نشان داد generic و SkylakeX با raw منطبق بودند؛ SkylakeX روی بعضی CPU runnerها غیرمجاز است، بنابراین اجرای رسمی به `OPENBLAS_CORETYPE=generic`, `OPENBLAS_NUM_THREADS=1`, `numpy==2.4.6`, و `torch==2.15.0+cpu` محدود شده است. روی سیستم دیگری که هش‌ها تطبیق نکنند، QA باید FAIL اعلام شود؛ raw به‌صورت خودکار جایگزین نمی‌شود.

**گیت اجرا مصرف شده است:** هیچ آموزش جدیدی در این مرحله مجاز نیست؛ فایل workflow پایلوت `if: false` دارد. گیت بعدی فقط ممیزی مستقل raw و تحلیل نتایج است.


## مرحله دوم: طرح قفل‌شده پیش از اجرا (آرشیو تاریخی؛ وضعیت اجرایی در ابتدای README به‌روزرسانی شده است)

**مأموریت جاری:** [Issue #81](https://github.com/mahhammad/My-Thesis/issues/81). مدیر مقرر کرده ممیزی **علمی نهایی** پس از گردآوری شواهد چندمولدی صورت گیرد. صدور **مجوز اجرای آموزش‌ها** منوط به **ممیزی فنی مقدماتی مستقل و مجوز کتبی جداگانه در GitHub** است. هر دو مجوز بعداً در Issue #81 صادر شد؛ این بند شرح وضعیت پیش از مجوز و اجراست. فایل `phase2_execution_gate.json` صورت تاریخی طرح را نشان می‌دهد. اجرای Test و مرج PR #79 نیز ممنوع‌اند.

- [phase2_preregistration.md](phase2_preregistration.md) — سؤال علمی، جدول `3 seeds × 2 families × 2 losses = 12 fits`، ۲۵ گام، β/λ/H، گرادیان و محدودیت تعمیم
- [phase2_inventory_reuse.csv](phase2_inventory_reuse.csv) — ۱۲ خانه هدف و وضعیت نامطمئن reuse تاریخی؛ هم‌نامی seed یا تشابه تعداد گام‌ها برای reuse کافی نیست
- [phase2_protocol.py](phase2_protocol.py) و [phase2_execution_gate.json](phase2_execution_gate.json) — قرارداد ثابت ۱۲ fit، گیت فنی/مدیریتی و ledger ضد تکرار
- [phase2_runner.py](phase2_runner.py) — dry-run پیش‌فرض؛ ۱۲ fit فقط با احراز رسمی مجوز، PASS همان SHA، opt-in و سقف سخت، بدون Test
- [phase2_frozen_data.py](phase2_frozen_data.py) — ساخت ورودی بدون آموزش و بازخوانی SHA-verified
- [inputs/phase2_train_validation/manifest.json](inputs/phase2_train_validation/manifest.json) — **سه Train/Validation منجمد**؛ هر seed دارای فایل NPZ و SHA256 مستقل، `n_train=300` و `n_validation=90`؛ [CI داده‌ای موفق #43622826092](https://github.com/mahhammad/My-Thesis/actions/runs/43622826092)
- [phase2_report.py](phase2_report.py) — محاسبه تمام COMMON metrics با `standard_metrics` از snapshot و **همان داده‌های منجمد**، بدون بازسازی وابسته به CPU یا افشای Validation labels هنگام inference
- [phase2_aggregate.py](phase2_aggregate.py) — پس از تکمیل ۱۲ raw، دلتای جفتی BA، flip count، اختلاف Recall، mean/SD سه seed؛ خروجی `PROVISIONAL` تا QA نهایی
- `test_phase2_*.py` و `.github/workflows/e-ch4-13-phase2-design.yml` — آزمون‌های بدون fit و CI مقید به NumPy/Torch و محیط BLAS؛ تأیید raw پایلوت **با هش فایل‌های اصلی**، بدون ادعای بازتولید بیت‌به‌بیت مولد تاریخی روی هر CPU

**محل مورد انتظار raw آموزش‌های آینده:** `results/phase2_300_25/` — **اکنون موجود است؛** دوازده آموزش مرحله دوم در اجرای رسمی #43670524935 تکمیل و داده‌ها به همان مسیر بایگانی شده‌اند. اجرای برنامهٔ dry-run:
```bash
python code/experiments/ch4_loss_controlled_comparison/phase2_runner.py
```
این دستور فقط فهرست ۱۲ کار را چاپ می‌کند، نه اینکه آموزش را آغاز کند. اجرای `--execute` با گیت فعلی PermissionError می‌دهد و نباید مجدداً اجرا شود؛ سقف ۱۲ آموزش مصوب مصرف شده است. تغییر در `tex/` یا خام پایلوت اعمال نشده است.
