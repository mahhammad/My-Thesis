# E-CH4-07 — ارزیابی مشاهده‌ای زیان حاشیه‌ای سیگموئیدی

**شناسه پایدار:** `E-CH4-07` | **گزارش:** [REPORT_FA.md](REPORT_FA.md) | **قرارداد:** [COMMON_METRICS_CONTRACT.json](../COMMON_METRICS_CONTRACT.json)

**وضعیت:** انتقال و تطبیق خروجی‌های تاریخی، بدون آموزش/گشت/دسترسی Test جدید؛ سه checkpoint مشاهده‌ای دارای گزارش عمومی نسخه ۲، در مقابل ۱۲ اجرای تاریخی آموزش که معیارهای عمومی ناقص دارند. بررسی نحوی TeX و اجرای Python unittest ویژه آزمایش در GitHub Actions با موفقیت تأیید شد؛ کامپایل کامل LaTeX در این مأموریت تأیید نشده است.

- `results/metrics_full.json`: خروجی عمومی جدا از تشخیص سیگموئید؛ سه وضعیت مشاهده‌ای از گزارشگر مشترک فصل ۴.۶؛ ۱۲ خلاصه آموزش تاریخی با علامت عدم تأیید مجدد.
- `results/sigmoid_validation_summary.csv`: ۱۸ سطر اختصاصی؛ سه checkpoint × شش پهنا؛ بایگانی بدون حذف در `archive/sigmoid_margin_diagnostics/`.
- `results/margin_training_summary.csv`: ۱۲ اجرای آموزش با هدف فقط سیگموئیدی؛ نسخه اصلی در `archive/margin_training/results/summary.csv`.
- `results/coverage.json`: وضعیت و دلیل داده‌های ناقص؛ `results/provenance.json`: منشأ و روش.
- `tables/*.tex`: جداول آماده ورود از `tex/chapter4.tex` با `\\input`.
- `analyze.py`: تشخیص اختصاصی و مقایسه چرخش رتبه برای τ؛ `test_analyze.py`: تست‌های سبک.

از ریشه مخزن: `python -m unittest discover -s code/experiments/sigmoid_margin_observational_evaluation -p 'test_*.py' -v`. بازساخت گزارش عمومی از داده خام با `code/experiments/benchmark_v1/report_snapshot.py` انجام می‌شود، نه `analyze.py`. داده خام checkpointهای مشاهده‌ای در `code/experiments/target_center_guidance_trajectory/results/representations_*.json` و بایگانی قدیمی محفوظ‌اند. **آمار ۷۶ اجرای تاریخی بازاجرا نشده‌اند.**

نمودار [حساسیت میانگین بر حسب τ](figures/sensitivity_vs_tau.svg) از داده تاریخی CSV ساخته شده و در محاسبه معیارهای عمومی دخالت ندارد. تست اختصاصی از طریق گردش‌کار [chapter4-e07-sigmoid.yml](../../.github/workflows/chapter4-e07-sigmoid.yml) روی GitHub Actions اجرا می‌شود.

**تأیید تست اختصاصی:** [GitHub Actions — موفق](https://github.com/mahhammad/My-Thesis/actions/runs/42775070311) (چهار unittest سبک؛ بدون Train/Test مجدد).
