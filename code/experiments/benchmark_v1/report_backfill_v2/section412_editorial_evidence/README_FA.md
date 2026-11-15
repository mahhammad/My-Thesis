# شواهد نگارشی §۴.۱۲ — Benchmark v1

> **مسیرهای معتبر پس از سبک‌سازی (۱۵ نوامبر ۲۰۲۶):** چهار ZIP و جدول‌های اصلی ۷۶ اجرای §۴.۱۲ در `code/experiments/standard_policy_comparison/results/` محفوظ‌اند. نتایج منحصربه‌فرد تاریخی و گزارش‌های مرحله‌ای R03 به `code/experiments/archive/benchmark_v1_legacy_evidence/` منتقل شده‌اند و Git blob اصلی آن‌ها تغییر نکرده است. اسکریپت‌های بازسازی از مسیر رسمی ZIP استفاده می‌کنند؛ گزارش‌های قدیمی زیر، وضعیت تاریخی زمان نگارش خود را ثبت می‌کنند و لزوماً مسیر فعلی فایل‌ها را نشان نمی‌دهند. راهنمای مسیرها در `code/experiments/archive/benchmark_v1_legacy_evidence/README_FA.md` آمده است.


**دامنه:** فقط مرجع عددهای §۴.۱۲ در `tex/chapter4.tex` در مأموریت `CH4-R03-SEC412-EDITORIAL-PR-01`. این فایل، متن پایان‌نامه نیست و برای ثبت provenance، نسخهٔ معیارها و تصمیم‌های علمی کنار آرشیو نگهداری می‌شود. هیچ آرشیو یا خروجی آزمایش بازنویسی نشده است.

## منابع ثابت

- شاخه مرجع فصل چهارم: `chapter4@de02fdda91e23db177de4429b3e1a9693c046d83`؛ فایل `tex/chapter4.tex` پیش از ویرایش Git blob `b341e38fcc44b9487a9e2f0ad7c2ac0a9a370637`.
- فهرست ۷۶ اجرا: `code/experiments/benchmark_v1/results/master_runs_76.csv`، Git blob `bcc2d7f4a4f39301973eb53c465ff05a6b4d0f21`. چهار ZIP آرشیو در `results/` ثابت هستند.
- ۱۱ معیار هندسهٔ مراکز: `code/experiments/benchmark_v1/report_backfill_v2/full_76_candidate/center_only_metrics.csv` در `experiment/ch4-metrics-backfill-v2@0a552e6e31999f0a9aeab499f62d3b54313c2e1c`، Git blob `4e7ab942ea6a825639829b868e41ed50d654d14d`.
- [رأی علمی مستقل `ACCEPT_QA76_CENTER_ONLY`](https://github.com/mahhammad/My-Thesis/issues/49#issuecomment-6796097800): ۷۶/۷۶ مراکز و ۸۳۶/۸۳۶ مقدار center-only پذیرفته شده‌اند؛ نه ۴۴ شناسهٔ دیگر.
- اطلاعات اعتبارسنجی aggregate تاریخی و منشأ: `experiment/ch4-metrics-p1-provisional-r03@d36ec6dc522e4e6d5f7ee4134f23c71f97b2aff3`، Git blob جدول `raw_aggregate_audit_76.json@d6d83e9937b58b68aa40cecc3994bf2542595a79`؛ `coverage_76x55_provisional.csv@41e2d2f3252fd5b3145c9215f6aee8d225ab9dea`.
- `code/experiments/COMMON_METRICS_CONTRACT.json` در base، blob `a7e1cea6276a339053a8ded2dfb56d2ad334f903`. فصل سوم منبع تعریف $T,BT,TNL,BTNL,I$، JS و زیان حاشیه است.
- [دستور محدودکنندهٔ ویرایش و Draft PR](https://github.com/mahhammad/My-Thesis/issues/49#issuecomment-6796729350).

## محل دقیق هر نتیجه

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

### قرارداد نسخه معیارها

- `center.T/BT/TNL/BTNL/I/pairwise_*` بر مبنای مراکز Train هستند؛ تأیید مستقل تنها همین ۱۱ شناسه را دربرمی‌گیرد.
- `train.R_JS` و `train.R_sqrt_JS` (و Validation دارای آغاز مجاز) از دادهٔ اصلی ثبت شده و **حساب aggregate کلاسی‌شان کنترل شده است**، اما محاسبهٔ JSD تک‌نمونه‌ای از checkpoint سیاست در این مأموریت انجام نشد. `R_sqrt_JS` یعنی میانگین ریشهٔ JSD در سطح نمونه، نه `sqrt(R_JS)`.
- ستون `R_{KL}^{legacy,clip}` در §۴.۱۲ عین `R_KL` ذخیره‌شدهٔ Benchmark اولیه است که در `code/experiments/benchmark_v1/metrics.py` clipping عددی دارد؛ **آن را به `R_KL_exact_support_aware` قرارداد v2 تبدیل نکنید**. اعداد صدکی/حاشیه/NLL و ties فقط با provenance و در جای مناسب خود گزارش شوند؛ QA76 آن‌ها را تأیید نکرده است.
- هر `node/node_class` برای نمونهٔ خارج از گراف سیاست شروع قراردادشده ندارد: ۲۶ اجرای مربوطه و همهٔ سنجه‌های Validation آن‌ها **not_applicable** هستند. اعداد Train به معنی قابلیت تعمیم نیستند.
- `experiment_specific.elapsed_s` در raw، زمان پایان آماده‌سازی خروجی `run.elapsed_s` است؛ `final.walltime_s` زمان آخرین history با گردکردن چهار رقم اعشار. اختلاف timestamp در ledger P1 باقی می‌ماند.
- `posthoc_score_backfill_76_compact.csv` اعداد NLL/Softplus شش‌رقمی دارد؛ نه calibrated Bayes posterior و نه independently verified-v2. متن آنها در بخش بعدی فصل بررسی می‌شود، نه در §۴.۱۲.

### انتقال جدول آزمون پیشین از متن اصلی به سند شواهد

این جدول تنها برای **حفظ منشأ نسخهٔ قبلی §۴.۱۲** کپی شده و **ورودی انتخاب سیاست، انتخاب tau، مقایسهٔ علمی جدید یا آزمون مستقل نیست**. مرجع قبلی `tex/chapter4.tex@de02fdda91e23db177de4429b3e1a9693c046d83`، جدول `tab:ch4-v1-historical-heldout`. هیچ دادهٔ Test دوباره خوانده یا اجرا نشده است.

| سیاست | Train BA % | Validation BA % | Test BA پیش‌تر گزارش‌شده % |
|---|---:|---:|---:|
| یکنواخت | 83.65 | 96.77 | 76.42 |
| حاشیه tau=.002 | 87.21 | 96.60 | 78.37 |
| حاشیه tau=.005 | 86.52 | 97.28 | 78.37 |
| حاشیه tau=.020 | 81.02 | 90.14 | 82.39 |
| پسین بیزی واقعی مولد | 84.32 | 97.45 | 83.88 |

**علت حذف از فصل:** این Test قبلاً مشاهده شده است و نباید برای نتیجه‌گیری درباره بهترین tau یا تأیید تعمیم تازه استفاده شود. نگهداری نسخهٔ قبلی و این جدول، امکان پیگیری تاریخی را حفظ می‌کند.

### تفکیک از مطالعات مجاور

نسخهٔ پیشین §۴.۱۲ همچنین به چهار اجرای بازیابی مراکز دست‌یافتنی و هفت اجرای بازارزیابی مراکز ثابت اشاره داشت؛ **این ۱۱ اجرا جزو ۷۶ اجرای Benchmark نیستند** و فایل‌های اصلی و گزارش آن‌ها در `code/experiments/benchmark_v1/RESULTS_FA.md` و زیرپوشه `results/` باقی مانده‌اند. مباحث حل معکوس و کران دست‌یافتنی در بخش‌های قبلی فصل چهارم بررسی می‌شوند. حذف جزئیات تکراری آن‌ها از §۴.۱۲ حذف داده یا تغییر نتیجه نیست.

**محدوده و کنترل:** فقط نوشتار §۴.۱۲ و این README جدید، بدون train، optimizer، inference، Test، تولید/تغییر مرکز، دستکاری P1/P3، تغییر قرارداد، تغییر فصل ۳، ویرایش بخش‌های ۴.۱۰/۴.۱۱/۴.۱۳ و بدون ادغام.
