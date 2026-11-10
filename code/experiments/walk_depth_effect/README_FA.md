# E-CH4-09 — اثر سقف واقعی طول گشت بر بازنمایی و طبقه‌بندی

**شاخه:** `experiment/ch4-depth-effect` از `chapter4@a49ea148cbe68d12d21b40c0c914f2e258928c8d`، پس از Merge شدن PR #43. **وضعیت اولیه:** منابع شناسایی‌شده؛ اجرای علمی منوط به عبور کنترل‌های منشأ و عددی است. **هیچ ادغامی مجاز نیست.**

## پرسش و تمایز از E08

این آزمایش **سقف واقعی H** را برای هر دو سیاست ثابت عوض می‌کند؛ در هر H ضرایب طول هندسی *سقف‌دار همان H* مستقلاً محاسبه می‌شوند. آزمایش E08 به‌جای آن، H=12 را ثابت نگه می‌دارد و گام‌های بعد از r را حذف و مجموع گام‌های نگه‌داشته‌شده را نرمال می‌کند. این دو شیوه هم‌ارز فرض نمی‌شوند.

- `β=0.4`، `λ=0.8`، `H∈{1,2,3,4,8,12}`، `float64`.
- سیاست `uniform`: همان logits صفر E08.
- سیاست `trained_shared`: checkpoint مشترک `margin_JSD, τ=.005, 50` گام تاریخی در E08؛ logits در طول تحلیل تغییر نمی‌کنند.
- فقط Train=300 و Validation=90، بذر اصلی `20261102`، Test ممنوع؛ برچسب Validation صرفاً پس از محاسبه توزیع‌ها برای سنجش.
- برای تمام Hها همان توزیع‌های گام‌به‌گام `Q_{i,t}` از E08 استفاده می‌شوند؛ هیچ انتقال جدیدی ساخته نمی‌شود.

## موجودی بازبینی پیش از اجرا

| منبع | محتوا | کنترل الزامی |
|---|---|---|
| `../exact_matrix_walk_validation/results/raw_step_distributions.npz` | توزیع‌های گام‌های ۱ تا ۱۲، آموزش/اعتبارسنجی هر دو سیاست | سلامت آرشیو، کلیدها، ابعاد، احتمال‌ها، جمع هر ردیف، مقایسه H=12 |
| `../exact_matrix_walk_validation/results/provenance.json` | بذر، ترتیب ۳۰۰ شناسه، SHA256 آرشیو checkpoint، هش‌های داده | ترتیب شناسه و هش برچسب‌ها و همخوانی دو snapshot |
| `../exact_matrix_walk_validation/results/snapshots/{uniform,trained_shared}/` | بازنمایی و مراکز H=12 و مشخصات دو سیاست | تطابق با محاسبه آف‌لاین H=12 تا خطای عددی |
| `../target_center_guidance_trajectory/results/representations_uniform.json` | برچسب‌های نمونه‌ای Train=300 / Validation=90 در ترتیب تاریخی | هش برچسب Train در E08، تعداد کلاس‌ها و همخوانی پیش‌بینی/مراکز H=12 |
| `../uniform_policy_hyperparameter_sensitivity/results/metrics_full.json` | مرجع مستقل ۴.۴ برای H=1,2,3,4,12، Uniform | مقایسه تمام معیارهای مشترک در عمق‌های مشترک |
| `../archive/bayes_center_inverse/results/depth_sensitivity_full300_and_subset72.json` | D10: تحلیل کران امکان دستیابی مراکز بیزی | صرفاً زمینه نظری؛ نه داده قابل‌ادغام با خروجی سیاست‌های ثابت |

**محدودیت داده:** E08 برچسب‌های نمونه‌ای را در فایل توزیع‌های خام ذخیره نکرده است؛ آرشیو `representations_uniform.json` منبع مستقل برچسب است و هم‌ترازی باید با هش مرجع و کنترل خروجی‌های E08 اثبات شود. هیچ `generated_data/*.npz` لازم نیست و هیچ مولدی فراخوانی نمی‌شود. اگر کنترل‌ها شکست بخورند، انتشار نتیجه متوقف می‌شود.

## اجرا / بازتولید بدون هزینه آزمایش اصلی

از ریشه مخزن:

```bash
python -m unittest discover -s code/experiments/walk_depth_effect -p 'test_*.py' -v
python code/experiments/walk_depth_effect/verify.py
python code/experiments/walk_depth_effect/analyze.py
```

`analyze.py` بدون `--execute` فقط برنامه را نشان می‌دهد. محاسبه نتایج صرفاً یک بار با `--execute` و فقط وقتی فایل نتیجه موجود نباشد مجاز است؛ هیچ آموزش، مونت‌کارلو، بازگشت به موتور گشت، بازتولید داده یا Test صورت نمی‌گیرد. خروجی‌های علمی پس از اجرا در `results/`، جدول‌ها و شکل‌ها در `tables/` و `figures/` و شرح در `REPORT_FA.md` ثبت می‌شوند.

روش سنجش: `benchmark_v1/report_snapshot.py::standard_metrics`، برای هر سیاست/H و هر تقسیم Train/Validation؛ NLLهای KL و JS جدا گزارش می‌شوند. زمان ثبت‌شده فقط **بازتحلیل آفلاین و تولید گزارش** است، نه زمان اجرای موتور گشت.
