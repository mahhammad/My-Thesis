# E-CH4-13 — مجوز اجرای پایلوت و ثبت گیت

**مدیر:** RESEARCH-MANAGER-04، Issue #78، [رأی رسمی](https://github.com/mahhammad/My-Thesis/issues/78#issuecomment-6880231186): ACCEPT_PREFLIGHT_DESIGN_WITH_CONDITIONS / CONDITIONAL_TWO_FIT_PILOT.

**Preflight evidence:** [CI #43614287838](https://github.com/mahhammad/My-Thesis/actions/runs/43614287838) PASS روی `fa65f37c82328c3b4178283ff1a17628a7a7b13b` و [CI #43614360524](https://github.com/mahhammad/My-Thesis/actions/runs/43614360524) PASS روی `f166d6a519b5440ec5ba1f065c1b459d733ebef1` با تست‌های مستقل، تست‌های canonical و dry-run بدون optimizer. در CIهای اولیه FAILهای واقعی fixture tie و نحوه فراخوانی تست ثبت شده و با اصلاح فقط در پوشه E-CH4-13 و workflow مربوط بسته شدند.

**حد ثابت اجرا:** فقط `data_seed=20261102`، `reference_seed=20261107`، `n=50`، ۳ کلاس، ۵ ویژگی، policy=shared، Adam `lr=.12`، logits صفر، `beta=.4,lambda=.8,H=12`، `float64` و دو objective `sigmoid_JS_margin_tau005` و `softplus_JS_tau005`، دقیقاً ۱۰ update برای هر کدام. هیچ انتخاب model/checkpoint از Validation و هیچ دسترسی Test مجاز نیست.

**Gate rerun:** workflow `.github/workflows/e-ch4-13-pilot.yml` با شناسه کامیت اجرا تنها پس از PASS تمام تست‌های سبک در همان workflow اجازه fit می‌دهد و حداکثر دو fit دارد. خروجی باید در `results/pilot_seed20261102_shared_two_fits/` شامل model.json، distributions.npz، metrics_full.json، trace.json، manifest.json، coverage.json و خلاصه CSV/SVG به صورت immutable ثبت شود. GitHub Actions اجازه مرج ندارد؛ PR #79 Draft باقی می‌ماند.

**گیت بعدی:** PILOT_READY_FOR_QA تنها با موفقیت واقعی CI، وجود raw و checksum، گزارش محدودیت و ممیزی مستقل. `18/24` fit، grid، full300، Test و مرج همچنان ممنوع.

**Checkpoint:** این سند مجوز آغاز خودکار همان پایلوت پس از عبور مجدد از CI است، نه ادعای اتمام یا پذیرش نتیجه.
