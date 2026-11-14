# E-CH4-08 — اعتبارسنجی مستقل محاسبه ماتریسی چندگامی

**وضعیت:** اجرا و آزمون‌های عددی موفق، ثبت نتایج خام روی `research/e08`؛ هنوز **ممیزی مستقل و ادغام انجام نشده‌اند**.

پیش‌ثبت علمی بدون بازنویسی در [PROTOCOL_FA.md](PROTOCOL_FA.md) و ژورنال پیش‌ثبت قرار دارد. تحلیل نهایی در [REPORT_FA.md](REPORT_FA.md)، گردش تصمیم‌ها و اصلاح پسینی شمارش انتخاب‌های افقی در [Research Journal](../../../research_journal/2026-11-10-e08-execution-and-delivery.md) است.

## نتایج و منبع حقیقت

- [`results/experiment_report.json`](results/experiment_report.json): تمام نتایج تخصصی انتشار، برش در شش عمق، MC و زمان‌ها.
- [`results/metrics_full.json`](results/metrics_full.json): ۲۴ بلوک کامل معیار مشترک برای ۲ سیاست × ۶ عمق × Train/Validation.
- [`results/raw_step_distributions.npz`](results/raw_step_distributions.npz): داده خام گام‌های ۱ تا ۱۲ برای هر دو سیاست، Train و Validation.
- [`results/provenance.json`](results/provenance.json): SHA256 checkpoint/آرشیو، بذر، ترتیب ۳۰۰ مرجع و مسیرها.
- [`results/coverage.json`](results/coverage.json): همه بودجه‌ها، بذرها، مبدأها و پوشش متریک.
- `results/snapshots/{uniform,trained_shared}/`: در هر پوشه `model.json`، `distributions.npz` و `report.json/report.md` مطابق `imc-snapshot-v2`.
- `tables/` و `figures/`: خروجی‌های TeX ساخته‌شده **از فایل JSON ثبت‌شده**، استفاده‌شده در بخش ۴.۸ `tex/chapter4.tex`.

بذر داده **20261102**؛ MC با **20261121** و **20261122**؛ `beta=.4, lambda=.8, H=12`. فقط ۳۰۰ Train و ۹۰ Validation؛ Test تاریخچه خوانده نشده است. سیاست‌های یکنواخت و checkpoint مشترک ذخیره‌شده بدون آموزش تازه ارزیابی شده‌اند.

## بازتولید بدون اجرای مجدد

از ریشه مخزن:

```bash
python -m unittest discover -s code/experiments/exact_matrix_walk_validation -p 'test_*.py' -v
python code/experiments/exact_matrix_walk_validation/verify_results.py
python code/experiments/exact_matrix_walk_validation/run.py
```

دستور آخر به‌صورت پیش‌فرض **dry run** است. اجرای محاسبه جدید با `--execute` فقط در صورت نبود فایل نتایج موجود مجاز است و در صورت وجود نتایج متوقف می‌شود؛ اینجا نیازی به تکرار نیست.

CI اختصاصی [E08 matrix walk](../../../.github/workflows/e08-matrix-walk.yml) با تست‌های سبک و بررسی read-only شواهد؛ در صورت موجود بودن `experiment_report.json` محاسبه جدید اجرا نمی‌کند. [اجرای موفق اصلی](https://github.com/mahhammad/My-Thesis/actions/runs/42804546810).

**یادداشت علمی:** شمار انتخاب‌های افقی در گراف باید خودحلقه مجاز ویژگی را در نظر بگیرد (`6334` روی مرجع کامل). گزارش اولیه `4968` فقط انتخاب‌های بین ویژگی‌های متفاوت را شمرد؛ این خطای *گزارش مشتق‌شده* بعداً بدون محاسبه مجدد تصحیح شد. مسیر و تصمیم در Research Journal محفوظ‌اند.
