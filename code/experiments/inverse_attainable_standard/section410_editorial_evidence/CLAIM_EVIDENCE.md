# ۴.۱۰ — دفتر شواهد نگارش (append-only)

**مأموریت:** CH4-R05-SEC410-EDITORIAL-PR-01  
**مبنای متن:** `chapter4@de02fdda91e23db177de4429b3e1a9693c046d83`  
**مطالعه اولیه:** `code/experiments/archive/multistep_policy_target_match/results` در commit بالا  
**B01-v2:** `experiment/ch4-inv01-standard-plan@968675cde8783adf69db13efd7de0fb3aab703c5`. نسخه اجرای موفق: `3aeb504505bfff8841417683dbaa509fed5d13c5`؛ commit ثبت raw: `f13f28d70a5ff90c8c894a5a2ced375637cdb316`؛ [CI تاریخی](https://github.com/mahhammad/My-Thesis/actions/runs/42986545002)، [پذیرش مستقل](https://github.com/mahhammad/My-Thesis/issues/49#issuecomment-6791688515).

**هش‌ها در جدول «Git blob SHA» هستند، نه ادعای SHA-256 مستقل محتوای فایل؛ هر مورد به مسیر زیر ref مربوط است.** تمام ارقام از raw/گزارش‌های پذیرفته‌شده استخراج و برای نگارش گردآوری شده‌اند؛ هیچ اجرای عددی جدید انجام نشده است.

## Claim → raw Git blob SHA → protocol → metric → safe wording

| Claim | Raw path و Git blob SHA | Protocol | Metric | Safe wording / محدودیت |
|---|---|---|---|---|
| C410-01 | `archive/multistep_policy_target_match/results/bootstrap_seed_20261107.json`: `f32756d06d28869e9ac3d333f29c03b29d964f1e`؛ `bootstrap_seed_20261108.json`: `b2f8faa4f74ca0b78187453a4de763d5f656bddb`؛ `bootstrap_seed_20261109.json`: `642255e06916159388caf65aa15acdf0272976ec` | نخست: سه subset از یک Train 300، ۷۲ مرجع در هر subset، ۲۴ نمونه/کلاس | ۹ مرکز هدف، ۲۷ اجرای روش | نه مسئله، **نه نه مولد مستقل** |
| C410-02 | `archive/multistep_policy_target_match/results/bootstrap_aggregate.json`: `ab806d64f817b324e1d50ae498ef8a4a12aba4c0` | نخست: Q45 / Adam90 / GN(45 Q + تا ۳۰ اصلاح در هر تلاش + restart مشروط) | JSD≤1e−8، موفقیت Q=0/9، Adam=5/9، GN=9/9 | فقط تحت همین بودجه و مرجع |
| C410-03 | همان `bootstrap_aggregate.json`: `ab806d64f817b324e1d50ae498ef8a4a12aba4c0` | نخست: CPU تک‌رشته‌ای؛ **final** JSD | mean final Q=0.00128656189037؛ Adam=0.00000211610172638؛ GN=4.94487263482e−9؛ مجموع wall Q=2.872859553s، Adam=26.584171676s، GN=10.769426562s | میانگین **نهایی**؛ نباید با میانگین best پروتکل دوم یکسان تلقی شود |
| C410-04 | همان `bootstrap_aggregate.json` و `bootstrap_seed_20261109.json`: `642255e06916159388caf65aa15acdf0272976ec` | نخست: GN method-inclusive | ۴۰۵ Q-grad، ۱۲۲ Jacobian، ۲۲۲ line-search، ۲۵۰ exact-forward، restart برای یک کلاس بذر 20261109 | شمارنده‌های بعضاً هم‌پوشان و غیرهم‌هزینه؛ هزینه تلاش ناموفق محفوظ |
| C410-05 | `archive/multistep_policy_target_match/BOOTSTRAP_REPORT_FA.md`: `3e434008c54d6a0e3df5f679f900c8928d3609ad` | نخست | RSS peak: **NOT_MEASURED** | عدم اندازه‌گیری را به‌جای عدد صفر گزارش نکنیم |
| C410-06 | `standard_plan/results/B01_v2/validated_run_01/manifest.json`: `d8440f7177721fec07b520362deb2552bde59a0e`؛ `paired_summary.json`: `171eccdc57c9e8ad3cc842261cf681898978ac1a` | دوم: یک مولد، ۱۵۰ Train متوازن ۵۰/کلاس، ۳ target × ۲ initialization × ۳ method | ۱۸ اجرا، ۳ مرکز هدف؛ data seed20261102/reference20261201/teacher20261211 | ۱۸ اجرای حلگر **نه** ۱۸ مسئله مستقل؛ witness از solver جدا |
| C410-07 | `standard_plan/results/B01_v2/validated_run_01/paired_summary.json`: `171eccdc57c9e8ad3cc842261cf681898978ac1a` و `summary.json`: `4928aee8c35d6ad3714718c1b8746da6480eba18` | دوم: Q100، Adam150، GN warm Q100 + تا ۳۰ GN و restart مشروط | success Q=0/6، Adam=5/6، GN=6/6؛ tol JSD≤1e−8 | شکست با cap گواه ناممکن بودن هدف نیست |
| C410-08 | `paired_summary.json`: `171eccdc57c9e8ad3cc842261cf681898978ac1a`؛ `trace.json`: `606c7069b67cdbee6e970a03961eface4c0d8b07` | دوم: انتخاب بهترین checkpoint، نگهداری جداگانه final | mean **best** JSD: Q=8.32657939192e−4؛ Adam=7.57229841793e−9؛ GN=1.84518848349e−9؛ شکست Adam در max best=1.3088990336e−8 | **best** با final study اول نباید با عنوان یک معیار یکسان تجمیع شود |
| C410-09 | `paired_summary.json`: `171eccdc57c9e8ad3cc842261cf681898978ac1a`؛ `summary.json`: `4928aee8c35d6ad3714718c1b8746da6480eba18` | دوم: plateau فقط diagnostic، توقف tolerance/cap/failure واقعی | صفر توقف plateau؛ ۶۰۰ Q-grad، ۵۴۰ exact-grad Adam، ۷۶ Jacobian GN، ۶ موفقیت GN بدون restart | «رفتار همین run»، نه نتیجه روش‌شناختی همگانی |
| C410-10 | `paired_summary.json`: `171eccdc57c9e8ad3cc842261cf681898978ac1a`؛ `summary.json`: `4928aee8c35d6ad3714718c1b8746da6480eba18`؛ `curve.csv`: `7b8b80b80ad88b2c0193cf90c12085721c37018b` | دوم: method-inclusive؛ GN حاوی Q warm مشترک با برچسب Q-only | wall روش‌های برچسب‌خورده Q=44.604572358s، Adam=44.112668984s، GN=52.612253389s؛ wall job=100.783917336s | جمع هزینه برچسب‌خورده **به‌دلیل overlap** زمان wall واقعی نیست |
| C410-11 | همان `paired_summary.json` و `summary.json` | دوم: پایش JSD دقیق در هر گام | Q-only monitor=606 forward/9.40635644s؛ Adam=546 forward/8.432350424s | هزینه monitor؛ «runtime بدون پایش» را از تفاضل ساده ادعا نکنیم |
| C410-12 | `summary.json`: `4928aee8c35d6ad3714718c1b8746da6480eba18` | دوم: عدد سطح کل فرایند | peak RSS=438.21875MiB؛ hard caps 900s/1536MiB، soft 840s/1430MiB | اوج کل فرایند، نه RSS هر method یا پیش‌بینی برای n300 |
| C410-13 | `manifest.json`: `d8440f7177721fec07b520362deb2552bde59a0e`؛ `standard_plan/b01_v2_run.py`: `1d0662f89c5cc152cc76647574424087aff65b58`؛ [QA](https://github.com/mahhammad/My-Thesis/issues/49#issuecomment-6791688515) | دوم: target حاصل از witness، در اختیار solve قرار نگرفته | وجود سیاست شاهد برای center | «حداقل یک سیاست مولد مرکز وجود دارد»، نه uniqueness/BA/secret رمزنگاری |
| C410-14 | aggregate pilot `ab806d64f817b324e1d50ae498ef8a4a12aba4c0` + paired دوم `171eccdc57c9e8ad3cc842261cf681898978ac1a` | پروتکل‌های جدا | اندازه ۷۲/۱۵۰؛ ۹ هدف در برابر ۳ هدف×۲ آغاز؛ mean final در برابر mean best، stop و budget متفاوت | برتری کلی، آزمون آماری pooled، تعمیم runtime به n300 یا بهبود classification ادعا نشود |

**دامنه مسیرها:** اختصار `archive/...` از `code/experiments/archive/...` و `standard_plan/...` از `code/experiments/inverse_attainable_standard/standard_plan/...` است. همه Git blob SHAهای بالا مربوط به رفرنس‌های قفل‌شده معرفی‌شده در ابتدای سند هستند.

## کنترل‌های نگارشی و آمادگی QA

- Labelهای موجود حفظ شدند: `sec:ch4-inverse-attainable`، `subsec:ch4-inverse-protocol`، `eq:ch4-inverse-success`، `subsec:ch4-inverse-algorithms`، `tab:ch4-inverse-bootstrap-cost`. Labelهای جدید فقط در بخش ۴.۱۰: `subsec:ch4-inverse-pilot-results`، `subsec:ch4-inverse-b01v2-results`، `tab:ch4-inverse-b01v2`، `subsec:ch4-inverse-interpretation`.
- ارجاع فصل سوم: `sec:ch3-inverse-center-policy` در Chapter 3 موجود است؛ تعریف معادله مسئله معکوس دوباره نقل نشده است.
- کنترل متنی: مرزهای section قبل و بعد از بخش ۴.۱۰ بدون تغییر و همه labelهای موجود حفظ؛ جدول‌ها هویت مستقل دارند.
- **XeLaTeX compile: NOT_CHECKED**؛ **PDF visual inspection: NOT_CHECKED**؛ **CI: NOT_CHECKED**. reviewer قبل از merge کنترل کند.
- Reviewer باید احتمال conflict با PR مستقل نگارش بخش ۴.۱۲ بر همان فایل `tex/chapter4.tex` را کنترل کند. هیچ ادغام یا ویرایش بخش‌های ۴.۱۱/۴.۱۲، verifier B02، raw قدیمی یا مسیر main/chapter4 انجام نشود.

**وضعیت: READY_FOR_EDITORIAL_AND_LATEX_QA**. این دفتر فقط شواهد نگارشی است، بدون graph، optimizer، solve، resource probe یا استفاده جدید از Test/Validation.
