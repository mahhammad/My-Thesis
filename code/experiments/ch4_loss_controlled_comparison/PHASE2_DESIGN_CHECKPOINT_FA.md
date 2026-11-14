# E-CH4-13 / RESEARCHER-06 — PHASE2_DESIGN_READY

**مأموریت:** [Issue #81](https://github.com/mahhammad/My-Thesis/issues/81) | **PR باز:** [#79 Draft](https://github.com/mahhammad/My-Thesis/pull/79) | **شاخه:** `experiment/ch4-loss-controlled-comparison-01` | **وضعیت:** `PHASE2_DESIGN_READY / NO_FIT / PRELIMINARY_TECHNICAL_QA_REQUIRED`.

## تصمیم مدیریتی جاری
ممیزی علمی نهایی E-CH4-13 تا جمع‌آوری شواهد چندمولدی به تعویق افتاده است. این تصمیم به معنای مجوز آموزش نیست. **ممیزی فنی مقدماتی مستقل، PASS واقعی کد مرحله دوم و اجازه کتبی اجرای آموزش جدید در GitHub باید جداگانه تعیین و ثبت شوند.** هنگام آماده‌سازی این گزارش Issue #80 باز و فاقد رأی ممیزی بود. فایل `phase2_execution_gate.json` با `training_authorized=false` قفل مانده؛ اجرای `--execute` فاقد مجوز است. حتی پس از مجوز، حداکثر ۱۲ fit؛ Test/merge/LaTeX ممنوع.

## بسته تحویل مرحله A
- `phase2_preregistration.md`: سؤال علمی، متغیرهای ثابت، کاهش balanced و خطر تفاوت scale، تجزیه تحلیل توصیفی سه seed
- `phase2_plan_manifest.json` و `phase2_protocol.py`: **۳ بذر × ۲ خانواده shared/class × ۲ تابع Sigmoid-JS/Softplus-JS = ۱۲ job**، full300، ۲۵ update، `τ=.005,β=.4,λ=.8,H=12,Adam lr=.12,float64`، init صفر، endpoint ثابت
- `phase2_inventory_reuse.csv`: دوازده خانه مطالعه و نامزدهای تاریخی؛ **هیچ نمونه تاریخی هنوز واجد شرایط reuse تأییدشده معرفی نشده است** چون هش raw، reference IDs، engine و شرایط عددی باید بررسی شوند
- `inputs/phase2_train_validation/manifest.json` و `ds<seed>_train_val.npz` برای seedهای `20261102/03/04`؛ فقط Train=300 + Validation=90، Test هرگز تولید نشده؛ **ساخت و راستی‌آزمایی داده بدون آموزش** در [GitHub Actions #43622826092](https://github.com/mahhammad/My-Thesis/actions/runs/43622826092)، کامیت بایتی `561a391d1c165075aee1d95a192c0bffcb4e2506`
- `phase2_frozen_data.py` و `phase2_data_preflight.py`: کنترل SHA فایل‌ها، هر آرایه و ماسک، کلاس‌ها، config و موارد عدم تطابق BLAS؛ همه fitها تنها از frozen NPZ خواهند خواند
- `phase2_runner.py`: dry-run، گیت fail-closed، استارت ۱۲ خانه ثابت، norm گرادیان/جهت و update، جلوگیری از تکرار، hash و snapshot raw، متریک canonical، ممنوعیت Test
- `phase2_report.py`: محاسبه کامل متریک مشترک از داده منجمد و snapshot با مراکز Train و `benchmark_v1/report_snapshot.standard_metrics`
- `phase2_aggregate.py`: تحلیل توصیفیِ جفتی BA/recall/flip/margin و mean/SD واقعی سه seed؛ تا قبل از همه ۱۲ raw با خطا متوقف می‌شود
- `test_phase2_*.py`، `.github/workflows/e-ch4-13-phase2-design.yml`: تست‌ها بدون fit، شامل derivative finite-difference، class hypotheses، leakage، ماسک، snapshot، KL، بودجه/گیت، raw integrity و data contract
- فایل‌های هشت‌گانه خام پایلوت و گزارش/خطاهای گذشته **بدون تغییر**؛ تثبیت SHA آنها در `test_phase2_pilot_unchanged.py`.

## شواهد واقعی CI و ضعف‌ها
- [CI اولیه مرحله دوم #43621841462](https://github.com/mahhammad/My-Thesis/actions/runs/43621841462) FAIL به‌علت عدم تطبیق fingerprint بازتولید تاریخی پایلوت در بعضی runnerها؛ خطا **به معنای تخریب snapshotها نیست**، و به شکل درستی متوقف شد
- [CI ۳۵ تست مرحله دوم #43621973697](https://github.com/mahhammad/My-Thesis/actions/runs/43621973697) PASS در runner سازگار با raw تاریخی
- [CI داده‌ای #43622826092](https://github.com/mahhammad/My-Thesis/actions/runs/43622826092) PASS پس از یک شکست اولیه import torch در freeze tool که با NumPy-only fingerprint رفع شد؛ بدون Test و بدون training
- [CI تست‌های جداشده مرحله دوم #43623071697](https://github.com/mahhammad/My-Thesis/actions/runs/43623071697) PASS، ۳۹ تست: ۱۷ phase2 + ۸ engine/pilot-fixture + ۴ canonical common metrics + ۷ snapshot pipeline + ۳ roundtrip. تست‌های phase2 طبق قرارداد raw frozen، SHA هر هشت فایل پایلوت، قرارداد گیت و no-leakage را راستی‌آزمایی کردند.
- بازتولید دوباره generator روی سخت‌افزار متفاوت ممکن است byte-exact نباشد؛ فایل‌های frozen ورودی واقعی مرحله دوم را قفل می‌کنند. **بازتولید بایت‌به‌بایت پایلوت تاریخی روی همه runnerها ادعا نمی‌شود**. ممیز فنی باید سازگاری داده‌های frozen با پروتکل و reuse تاریخی را مستقل بررسی کند.
- همه fitهای phase2، معیارهای عددی phase2، نمودارهای واقعی، QA علمی نهایی، Test و XeLaTeX = `NOT_RUN`.

## محل تداوم برای جانشین
شاخه جدید نسازد و خام پایلوت و `inputs/phase2_train_validation` را تغییر ندهد. ابتدا Issue #81 و مجوز رسمی فنی، سپس `phase2_execution_gate.json`، `phase2_protocol.py`، `phase2_plan_manifest.json` و گزارش CI روی همان commit را بررسی کند. تا مجوز کتبی مشخص **هیچ fit** اجرا نشود. در صورت هر FAIL یا عدم تطابق SHA، STOP و checkpoint در Issue.

**درخواست گیت بعدی:** ممیزی فنی مقدماتی مستقل از پژوهشگر ۰۶ و سپس تصمیم جداگانه مدیر درباره اجازه حداکثر ۱۲ fit. این تحویل، **مجوز خودکار آموزش** یا پذیرش علمی نتیجه نیست.
