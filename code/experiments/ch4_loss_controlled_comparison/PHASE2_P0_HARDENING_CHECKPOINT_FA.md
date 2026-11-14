# تحویل اصلاحات P0 مرحله دوم — E-CH4-13 / RESEARCHER-06

**مرجع مدیریتی:** [Issue #81 / TECHNICAL_PREFLIGHT_ACCEPT_WITH_FIXES](https://github.com/mahhammad/My-Thesis/issues/81#issuecomment-6883629540)  
**شاخه:** `experiment/ch4-loss-controlled-comparison-01`؛ PR #79 همچنان Draft؛ `main` و `tex/` بدون تغییر.  
**وضعیت اجرای واقعی:** `NO_FIT / TECHNICAL_FIXES_IMPLEMENTED / AWAITING_MANAGER_REVIEW_AND_EXPLICIT_EXECUTION_AUTHORIZATION`.  
**ممیزی علمی نهایی:** پس از نتایج چندمولدی. **هیچ برنامه‌ای به‌صورت خودکار آموزشی اجرا نمی‌کند.**

## P0-1 — گیت مجوز رسمی و یک‌بارمصرف
- [`phase2_github_gate.py`](phase2_github_gate.py): با GitHub token واقعی، head شاخه و SHA دقیق، کامنت تأیید اصلاحات فنی `TECHNICAL_PREFLIGHT_FIXES_ACCEPTED` و کامنت **جداگانه متأخر** `PHASE2_EXECUTION_AUTHORIZED` از صاحب مخزن در Issue #81 را می‌خواند. هر دو باید `candidate_sha=<exact40hex>` و `max_fits=12` *یکتا* داشته باشند. نتیجه CI طراحی باید `conclusion=success` روی همان SHA در workflow `e-ch4-13-phase2-design.yml` باشد. pagination ناشناخته، SHA/branch نامعتبر، کامنت جعلی/متناقض و CI ناموفق STOP.
- [`e-ch4-13-phase2-authorized.yml`](../../../.github/workflows/e-ch4-13-phase2-authorized.yml): فقط `workflow_dispatch` از شاخه مصوب و با اعتراف صریح کاربر؛ هیچ `push` trigger ندارد. قبل از fit مجوز را **آنلاین** بررسی و بر اساس SHA، به‌شکل اتمیک یک Git tag/ref `refs/tags/e-ch4-13-phase2-budget-spent-<SHA>` ایجاد می‌کند. اگر قبلاً وجود داشته باشد GitHub 422 و STOP. این نشانگر با شکست CI پاک نمی‌شود. concurrency تک‌گروهی، `cancel-in-progress: false`، اجرای غیرمجاز متوقف.
- `phase2_runner.py`: علاوه بر بررسی gate موقت (فقط در runtime و بدون commit)، خواندن مجدد رأی آنلاین/CI و وجود SHA marker از GitHub؛ تطبیق `GITHUB_RUN_ID`، branch/workflow، runtime pinned، رمز تصادفی یک‌بارمصرف، خروجی تازه. JSON gate پیش‌فرضِ committed همچنان BLOCKED است. توکن داخل raw/provenance/artifact نوشته نمی‌شود.
- **قید مهم:** فایل workflow مستقیماً در گیت واقعی اجرا نشده است، چون مجوز رسمی هنوز وجود ندارد. آزمون‌های تستِ GitHub با mock/fixture هستند، نه شاهد اجرای مجاز ۱۲ fit؛ تأیید عملکرد واقعی آنلاین پس از صدور مجوز باقی می‌ماند.

## P0-2 — شکست میانی و بودجه
- برای هر یک از **حداکثر ۱۲** خانه `attempt_charges/<run>.json` و `budget_checkpoints/attempt_NN.json` **پیش از ورود به optimizer** با حالت immutable ایجاد می‌شود.
- `optimizer_checkpoints/<run>/step_XX.json` همه گام‌های تکمیل‌شده شامل logits و trace را ثبت می‌کند؛ `optimizer_updates_applied_to_stored_logits` مستقل از `evaluated_at_updates` است. این checkpointها **فقط شواهد** هستند و مجوز resume/retry نیستند.
- پس از fit، `trace.json` قبل از snapshot/report ذخیره می‌شود. snapshot/metric/coverage/evidence هر خانه جداست؛ شکست optimizer یا report به `failures/` ثبت و استثناء دوباره صادر می‌شود؛ manifest نهایی فقط پس از کامل‌شدن ۱۲ خانه. وجود حتی خروجی ناقص مانع rerun می‌شود.
- در workflow حفاظت‌شده، `actions/upload-artifact@v4` با `if: always()` و retention=90 روز **تمام نتایج ناقص و کامل** را حتی هنگام CI FAIL ذخیره می‌کند. نشانگر مصرف GitHub مستقل از طول عمر artifact باقی می‌ماند. این نگهداری ۹۰روزه دائمی نیست و باید هنگام تحویل نتایج در مخزن بایگانی نهایی انجام شود.

## P0-3 — تست کامل مصنوعی، بدون optimizer واقعی
[`test_phase2_hardening.py`](test_phase2_hardening.py) موارد زیر را در fixture پوشش می‌دهد: تأیید آنلاین صحیح، رد فقدان رأی/صاحب غیرمجاز/چند SHA/ترتیب غلط، عدم موفقیت CI، شاخه و HEAD غلط، تکرار نشانگر یک‌بارمصرف، عدم ایجاد gate هنگام خطای اعتبارسنجی، تزریق شکست fit و باقی‌ماندن budget و failure files، رد fingerprint تغییرکرده قبل از هر optimizer، و **اجرای mock تمام ۱۲ خانه از gate تا raw/coverage/CSV/manifest**. آزمون مثبت تجمیع ۱۲ خانه در `phase2_aggregate.py` جداگانه با fixture عددی و ۶ paired comparison اجرا شده است. تست‌ها فقط mock هستند؛ هیچ optimizer جدیدی اجرا نشده و Test خوانده نشده است.

## تاریخچه CI، خطاها و وضعیت
- قبل از P0: [CI #43623484023](https://github.com/mahhammad/My-Thesis/actions/runs/43623484023): ۴۰ PASS روی `970eba66`.
- طی اصلاحات، [CI #43637454103](https://github.com/mahhammad/My-Thesis/actions/runs/43637454103) FAIL روی fixture گیت جدید (SHA مبهم، کنترل duplicate)؛ مورد اصلاح شد.
- [CI #43637661792](https://github.com/mahhammad/My-Thesis/actions/runs/43637661792) FAIL روی mock snapshot که برخلاف `save_snapshot` اصلی، پوشه‌ای حاوی trace را ممنوع می‌کرد؛ fixture با رفتار canonical منطبق شد.
- [CI #43637787275](https://github.com/mahhammad/My-Thesis/actions/runs/43637787275) روی `aa0dcecc21a832d2edb8a6fc0cb48ac6965e347e`: **۵۰ آزمون PASS**؛ ۲۸ تست phase2 + ۸ engine + ۴ common metrics + ۷ snapshot + ۳ roundtrip. کد پس از این SHA فقط برای افزودن ستون نهایی CSV موجودی reuse و تهیه همین گزارش تغییر یافته است؛ روی SHA تازه نیز CI باید مستقل PASS شود.
- CI قدیمی پایلوت [#43637342310](https://github.com/mahhammad/My-Thesis/actions/runs/43637342310) در تشخیص fingerprint تولید مجدد با BLAS جدید FAIL شد؛ نه انکار و نه جایگزین موفقیت CI فاز۲ است. داده‌های اصلی n=300 از frozen NPZ تاییدشده خوانده می‌شوند؛ داده خام پایلوت ۲ اجرا و تاریخچه خطاها دست‌نخورده مانده‌اند.

## وضعیت reuse و تصمیم ایمن
[`phase2_inventory_reuse.csv`](phase2_inventory_reuse.csv) تمام ۱۲ خانه را پوشش می‌دهد؛ هر مورد تاریخی `NOT_COMPARABLE_FOR_REUSE` **در معنای نبود شواهد کافی برای اثبات برابری بایتیِ داده/پروتکل/کد** است، نه ادعای اثبات تفاوت عددی. هیچ خروجی تاریخی بی‌تطبیق به phase2 الصاق نمی‌شود. بودجه جدید حداکثر ۱۲ fit است، اما بدون رأی/اجازه **صفر**.

## درخواست رأی مدیر
گیت اصلاحات فنی `P0_FIXES_SUBMITTED_FOR_MANAGER_REVIEW`؛ پس از CI نهاییِ SHA جاری، مدیر می‌تواند این سه اصلاح را مستقل بازبینی کند. برای فعال‌سازی، دو رأی رسمی متوالی صاحب مخزن در **Issue #81** با markerها و `candidate_sha` قفل‌شده لازم‌اند. تا ثبت آن‌ها `NO_NEW_FIT` قطعی است. اجرای واقعی این workflow و پذیرفته‌شدن علمی نتایج هنوز `NOT_RUN / NOT_VERIFIED` است؛ PR #79 Draft/بدون merge.
