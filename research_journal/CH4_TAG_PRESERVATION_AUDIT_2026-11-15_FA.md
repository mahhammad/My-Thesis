# ممیزی تطبیقی نگهداشت آزمایش‌های فصل چهارم با تگ پیش از پاک‌سازی

**شناسه:** CH4-TAG-CONSERVATION-2026-11-15-01  
**مبنای تاریخی:** [`backup/pre-cleanup-2026-11-15`](https://github.com/mahhammad/My-Thesis/releases/tag/backup%2Fpre-cleanup-2026-11-15) → `a8c9027460e538fa04b8aab2f762eb4ba4c31e3d`  
**شاخهٔ ممیزی:** `editorial/experiment-report-consolidation-01`، پایهٔ بررسی `fb6a1933036f1573d15c4511d115f042edb54d6b`  
**نوع ممیزی:** مقایسهٔ ساختار Git و هویت Git blob، راستی‌آزمایی مسیر سندها و مستندسازی نحوهٔ اجرا و انبارش. **هیچ آموزش، اجرای حلگر، محاسبهٔ تازهٔ مراکز یا بازنویسی داده انجام نشده است.**

## ۱. نتیجهٔ ساختاری

در `code/experiments/` **۱۱ پوشهٔ مطالعهٔ رسمی §۴.۳ تا §۴.۱۳** و یک پوشهٔ `benchmark_v1` برای کد و موتورهای مشترک وجود دارد. گزارش مستقل هر مطالعه در `README_FA.md` خودش است؛ فایل‌ها به‌صورت `results/`، `tables/`، `figures/` و ورودی/کد اختصاصی همان آزمایش نگهداری می‌شوند. تفاوت میان زیرساخت مشترک Benchmark و آزمایش §۴.۱۲ یعنی `standard_policy_comparison` در نمایه اصلی ثبت شده است.

| بخش | پوشه | فایل غیر-Markdown در تگ | بایت‌یکسان در شاخه | Python با تغییر نسخه | فایل غیر-Markdown حذف‌شده | فایل غیر-Markdown افزوده | نتایج زیر `results/` در شاخه |
|---|---|---:|---:|---:|---:|---:|---:|
| §۴.3 | [`q-validation`](../code/experiments/q-validation/README_FA.md) | 16 | 16 | 0 | ۰ | 0 | 7 |
| §۴.4 | [`uniform-hyperparameter-sensitivity`](../code/experiments/uniform-hyperparameter-sensitivity/README_FA.md) | 16 | 16 | 0 | ۰ | 0 | 6 |
| §۴.5 | [`guided_informative_uniform_mixture`](../code/experiments/guided_informative_uniform_mixture/README_FA.md) | 33 | 33 | 0 | ۰ | 0 | 7 |
| §۴.6 | [`target_center_guidance_trajectory`](../code/experiments/target_center_guidance_trajectory/README_FA.md) | 42 | 41 | 1 | ۰ | 0 | 15 |
| §۴.7 | [`sigmoid_margin_observational_evaluation`](../code/experiments/sigmoid_margin_observational_evaluation/README_FA.md) | 23 | 20 | 3 | ۰ | 0 | 10 |
| §۴.8 | [`walk_depth_effect`](../code/experiments/walk_depth_effect/README_FA.md) | 32 | 24 | 8 | ۰ | 0 | 12 |
| §۴.9 | [`exact_matrix_walk_validation`](../code/experiments/exact_matrix_walk_validation/README_FA.md) | 21 | 18 | 3 | ۰ | 0 | 11 |
| §۴.10 | [`inverse_attainable_standard`](../code/experiments/inverse_attainable_standard/README_FA.md) | 761 | 750 | 11 | ۰ | 0 | 669 |
| §۴.11 | [`bayes_center_reachability_standard`](../code/experiments/bayes_center_reachability_standard/README_FA.md) | 14 | 14 | 0 | ۰ | 0 | 5 |
| §۴.12 | [`standard_policy_comparison`](../code/experiments/standard_policy_comparison/README_FA.md) | 15 | 15 | 0 | ۰ | 3 | 13 |
| §۴.13 | [`ch4_loss_controlled_comparison`](../code/experiments/ch4_loss_controlled_comparison/README_FA.md) | 566 | 566 | 0 | ۰ | 0 | 440 |
| **جمع** | **۱۱ مطالعه** | **1539** | **1513** | **26** | **۰** | **3** | **1195** |

در مجموع از **1539 فایل غیر-Markdown** این ۱۱ آزمایش در تگ، **1513 فایل** Git blob دقیقاً یکسان دارند؛ **26 فایل تغییریافته همگی Python** هستند و هیچ فایل داده/نتیجه/جدول/تصویر/آرشیو خام حذف یا تغییر هویت نداده است. **3 فایل غیر-Markdown جدید** به مجموعهٔ §۴.۱۲ افزوده شده است. تغییر Python از تگ به معنای لزوماً تغییر نتیجه نیست؛ این ممیزی، هم‌ارزی محاسباتی تمام کدهای تغییریافته را از نو اثبات نکرده است.

## ۲. تفکیک اسناد تجمیع‌شده از نتیجهٔ خام

از میان همان ۱۱ پوشه، 48 فایل Markdown تاریخی از checkout فعال برداشته شده‌اند. محتوای گزارش‌ها در `README_FA.md`های مطالعات تجمیع شده و اصل هر فایل با `git show a8c9027460e538fa04b8aab2f762eb4ba4c31e3d:<old-path>` از تگ قابل دستیابی است. این گزارش **حفظ دقیق بایت همهٔ عبارت‌های متن حذف‌شده را گواهی نمی‌کند**؛ مقایسهٔ متنی/ویرایشی با ممیزی نگارشی جداست. برای هر ۱۱ README فعلی، یک پیوست «اجرای تاریخی و نقشه ذخیره‌سازی» افزوده شده که ورودی، کد اجرایی، محل raw/results، فایل‌های مشتق و رفتار نوشتن خروجی را به‌صورت موردی شرح می‌دهد.

## ۳. کنترل نمونه‌های حساس با Git blob

| شاهد | نتیجه |
|---|---|
| §۴.۱۲ / `core_50_16runs.zip` | Git blob مشترک و نسخهٔ فعلی: `5b8136dce91b605b0cc265861a903a604e86e453` |
| §۴.۱۲ / `extra_25_12runs.zip` | Git blob مشترک و نسخهٔ فعلی: `5ee0100e9be728acac10add788e204d13f3723e9` |
| §۴.۱۲ / `size_25_24runs_seed20261102.zip` | Git blob مشترک و نسخهٔ فعلی: `401714cbfb01cb01aad16fa4a736752edd9833b5` |
| §۴.۱۲ / `size_25_24runs_seeds20261103_20261104.zip` | Git blob مشترک و نسخهٔ فعلی: `bdf54142ec5cde3c1e74e7e00c3cff747f00882f` |
| §۴.۱۰ / `runner_v3/data_contract.py` قفل‌شده | در هر دو نسخه `4984961c6a8fd8057a7c1ff3d292d30ad6c9afcb` |
| §۴.۱۰ / `results/metrics_full.json` و `results/coverage.json` | هر دو دقیقاً با blob تگ برابر |
| §۴.۱۱ / `raw/section411_standard_S_103_audit_package.zip` | blob دقیقاً برابر با تگ |
| §۴.۱۳ / `results/phase2_300_25/manifest.json` | blob دقیقاً برابر با تگ |

**مسیرهای قبلی موتور مشترک:** `StreamGraph`، `ExactGraph` و `bootstrap_inverse.py` به `code/experiments/benchmark_v1/` منتقل شده‌اند؛ هویت Git blob کد اصلی ثابت است. کدهای `observe.py` و `projection.py` نیز همان‌جا مستقر هستند. گیت‌ها و trust-root تاریخی B02 را برای تأیید مسیر جدید ویرایش نکرده‌ایم؛ آن ارجاعات فقط در worktree تاریخیِ منطبق با قرارداد قدیمی معتبرند.

## ۴. آرشیو بازنشسته و محدودیت بازپخش

پوشه `code/experiments/archive/` در **تگ** وجود دارد و از checkout فعال حذف شده است؛ این کار به معنای حذف شواهد از Git نیست. راهنمای [بازیابی آرشیو تاریخی](archive_recovery/README_FA.md) و نمایهٔ ۱۳۷ فایل `archive_recovery/ARCHIVE_RECOVERY_2026-11-15.json` باید برای استخراج ساختار اولیه استفاده شوند. اسکریپت‌هایی که هنوز مسیر `archive/` را انتظار دارند (برای نمونه برخی بازپخش‌های تاریخی §۴.۷) برای اجرا به checkout تگ یا بستهٔ بازیابی نیاز دارند. **وجود پرونده‌های رسمی فعلی با قابلیت بازاجرای تمام اسکریپت‌های تاریخی در checkout سبک، دو ادعای جداگانه‌اند.**

## ۵. قرارداد نگهداشت و بازتولید امن

- گزارش جامع `README_FA.md` هر آزمایش نه حذف و نه با اجرای گزارشگر جایگزین شود؛ ایجاد یک `REPORT_FA.md` یا فایل مشتق **جدید** در کنار آن مجاز است.
- فایل‌های `results/` رسمی مرجع ممیزی و provenance هستند. هر برنامه‌ای که خودکار به این مسیر می‌نویسد، فقط با مقصد خروجی مجزا یا در worktree جدید بازاجرا شود.
- هیچ ادعای تازه دربارهٔ برتری روش، تساوی موتورهای متفاوت یا تأیید کامل اجرای تاریخی از این مقایسهٔ فایل‌ها استنتاج نمی‌شود.
- تا پایان ممیزی نگارشی/علمی نهایی، [PR #86](https://github.com/mahhammad/My-Thesis/pull/86) در وضعیت Draft بماند و `main` دست‌نخورده حفظ شود.

**حکم ممیزی نگهداشت:** `PASS_FILE_PRESERVATION_11_STUDIES` برای وجود و هویت بایتی شواهد غیرمتنی ۱۱ آزمایش؛ `LIMITED_REPLAY_COMPATIBILITY` برای بازاجرای کدهای تاریخی که آرشیو جدا لازم دارند؛ `EDITORIAL_REPORT_REVIEW_PENDING` برای بازبینی نهایی متن‌های بسیار طولانی. این رأی، گواه جدید علمی آزمایش‌ها نیست.
