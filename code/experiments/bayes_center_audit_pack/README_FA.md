# RESEARCHER-02 — CH4-D10D11-EVIDENCE-PACK-01

**مرجع قفل:** `chapter4@de02fdda91e23db177de4429b3e1a9693c046d83`؛ تاریخ ۱۱ نوامبر ۲۰۲۶؛ Issue #49 / دستور `6791595363`.  
**قرارداد معیار:** `chapter4-common-evaluation-contract-v2`، کد مرجع `code/policy_evaluator/metrics.py:167–187`.  
**وضعیت:** آماده ممیزی مستقل؛ هیچ تغییر مدل/داده/لاتک یا اجرای حلگر/بازپخش وجود ندارد.

## راهنمای فایل‌ها
- [results/d10_coordinate_certificate.csv](./results/d10_coordinate_certificate.csv): ۲۲۲ گواه D10 = ۳۷ ترکیب × ۳ کلاس × ۲ نوع کران؛ شناسه class/config، upper/target/gap و blob SHA.
- [results/d10_certificate_summary.json](./results/d10_certificate_summary.json): شبکه، قرارداد آزادسازی، مثبت‌بودن همه ۱۱۱ گواه full300 و ۱۱۱ گواه subset72.
- [results/d11_coordinate_and_optimizer_evidence.json](./results/d11_coordinate_and_optimizer_evidence.json): ۱۸ قید مؤلفه‌ای، دو گواه equal، کران پایین JSD، ۶ ردگیری GN، جدایی شکست حلگر و ناممکن‌بودن.
- [results/center_geometry_metrics.json](./results/center_geometry_metrics.json): ۱۸ ماتریس مرکز تحقق‌یافته از داده ذخیره‌شده (۱۵ D10، سه D11)، T/BT/TNL/BTNL/I و ماتریس‌های JS و √JS، version و coverage per-model.
- [results/metric_coverage.json](./results/metric_coverage.json): منشأ، status/reason، محدودیت KL تاریخی و نبود R_sqrt_JS.
- [results/static_validation_checks.json](./results/static_validation_checks.json): assertionهای سبک خواندن JSON/ZIP و تطبیق مقادیر؛ نه اجرای تست پایتون.
- [EDITING_HANDOFF_FA.md](./EDITING_HANDOFF_FA.md): دو پاراگراف گذار، جدول‌های چاپی، اصلاحات موضعی §۴.۱۱، ادعاهای مجاز/غیرمجاز.

## Provenance — source blob SHA
- `grid` — `code/experiments/archive/bayes_center_inverse/results/depth_sensitivity_full300_and_subset72.json` — `1c0e1e393361742ad7ede179d00c6247d86da75f` — [source](https://github.com/mahhammad/My-Thesis/blob/de02fdda91e23db177de4429b3e1a9693c046d83/code/experiments/archive/bayes_center_inverse/results/depth_sensitivity_full300_and_subset72.json)
- `bound` — `code/experiments/archive/bayes_center_inverse/results/full_reference_bounds.json` — `43d55261ccdf3ffd007c933c5fbc4bef59f587d3` — [source](https://github.com/mahhammad/My-Thesis/blob/de02fdda91e23db177de4429b3e1a9693c046d83/code/experiments/archive/bayes_center_inverse/results/full_reference_bounds.json)
- `compact` — `code/experiments/archive/bayes_center_inverse/results/full300_maxdiag_compact_summary.json` — `37431cc4b0eb3603260940c9514deab3861c1974` — [source](https://github.com/mahhammad/My-Thesis/blob/de02fdda91e23db177de4429b3e1a9693c046d83/code/experiments/archive/bayes_center_inverse/results/full300_maxdiag_compact_summary.json)
- `summary` — `code/experiments/archive/bayes_center_inverse/results/summary.json` — `3309103d2449d643827cafd18efd4819a6ebd584` — [source](https://github.com/mahhammad/My-Thesis/blob/de02fdda91e23db177de4429b3e1a9693c046d83/code/experiments/archive/bayes_center_inverse/results/summary.json)
- `bayes7` — `code/experiments/archive/bayes_center_inverse/results/bayes_seed_20261107_gn10.json` — `af52d91ff3096666cdc636dc3f07f405ec515092` — [source](https://github.com/mahhammad/My-Thesis/blob/de02fdda91e23db177de4429b3e1a9693c046d83/code/experiments/archive/bayes_center_inverse/results/bayes_seed_20261107_gn10.json)
- `bayes8` — `code/experiments/archive/bayes_center_inverse/results/bayes_seed_20261108.json` — `2d5f2805c15c6b611fe2e7cdd753b81c513b23ba` — [source](https://github.com/mahhammad/My-Thesis/blob/de02fdda91e23db177de4429b3e1a9693c046d83/code/experiments/archive/bayes_center_inverse/results/bayes_seed_20261108.json)
- `bayes9` — `code/experiments/archive/bayes_center_inverse/results/bayes_seed_20261109_gn10.json` — `74c36046fef1d63b3bb79401d988616e79d8b7e3` — [source](https://github.com/mahhammad/My-Thesis/blob/de02fdda91e23db177de4429b3e1a9693c046d83/code/experiments/archive/bayes_center_inverse/results/bayes_seed_20261109_gn10.json)
- `zip` — `code/experiments/archive/historical_archive/archives/full300_maxdiag_bootstrap_experiment.zip` — `c85dc00fc5ea23bffb45481261b3e4d32dd9e659` — [source](https://github.com/mahhammad/My-Thesis/blob/de02fdda91e23db177de4429b3e1a9693c046d83/code/experiments/archive/historical_archive/archives/full300_maxdiag_bootstrap_experiment.zip)

کد تاریخی `feasibility_bound.py` در `code/experiments/archive/bayes_center_inverse/` کران مؤلفه‌ای را تحت آزادسازی سیاست محاسبه می‌کند. `full300` بازگشت مبدأ را مجاز کرده؛ `subset72` مبدأ را حذف می‌کند. این‌ها **necessary outer bounds** هستند. نقض آن‌ها اثبات عدم دستیابی تحت سیاست محدود است، اما قرار گرفتن درون جعبه کران‌ها وجود یک سیاست ثابت را اثبات نمی‌کند.

ZIP آرشیوی D11 دارای **۲۰ عضو**، ازجمله شش فایل `*_class{0,1,2}.json`، دو summary و Uniform است. داده JSONهای درون ZIP بدون re-run استخراج شدند. `equal` دو نقض و کران JSD مثبت دارد؛ `bounded_equal` هیچ نقض تک‌مؤلفه‌ای ندارد، با این حال هر شش trace فقط `not_recovered_within_budget` هستند.

R_KL قدیمی با log-clamp 1e-300 نسخه‌گذاری شده؛ R_sqrt_JS بدون نمونه‌های فردی `not_measured` است و از sqrt(mean JS) جایگزین نشده. اعداد هندسه مرکز از ماتریس‌های ۳×۳ ذخیره‌شده و برابر وزن کلاس‌ها بازحساب و با I، میانگین JS و min √JS آرشیو کنترل شده‌اند.

**محدودیت:** شواهد حفظ‌شده برابر اثبات صحت مستقل implementation بلمن یا اجرای تست‌های منبع نیست؛ این دو نیازمند ممیز مستقل‌اند. هیچ فایل `main`، `chapter4`، آرشیو یا `tex/` تغییر نمی‌کند.
