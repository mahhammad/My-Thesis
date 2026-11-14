# E-CH4-13 — RECOVERY_CHECKPOINT_01

**وضعیت:** PILOT_REPORTING_FAILURE / RAW_PRESERVED / NO_RETRAINING.

- گیت قبل از پایلوت: [GitHub Actions run 43614497613](https://github.com/mahhammad/My-Thesis/actions/runs/43614497613)، job guard-and-test **SUCCESS**: ۲۲ آزمون PASS.
- job اجرا: **FAIL** پس از دو fit مصوب، در ساخت `coverage.json`، به‌دلیل `TypeError: float() argument must be a string or a real number, not 'dict'` در تبدیل `margin_quantiles` که نوع آن dictionary است. این خطا در معیار اجرای آموزش و گرادیان گزارش نشده؛ جای دقیق خطا مرحله *post-fit reporting* بوده است.
- artifact سالم‌شده پیش از خطای گزارش: [raw artifact 14121258818](https://github.com/mahhammad/My-Thesis/actions/runs/43614497613/artifacts/14121258818)، ۸ فایل خروجی: `model.json`، `distributions.npz`، `metrics_full.json`، `trace.json` برای هر یک از دو هدف. Artifact SHA256 ZIP در لاگ: `d243b375f4b0034d31069760dbc0606dfbd93ea91ca28965a732d2c6708ee0d1`.
- طبق سقف دو fit مدیریتی، **هیچ training دیگری مجاز نیست**. فقط بازیابی raw از همان artifact، اعتبارسنجی هش/داده/فایل‌های مدل و بازساخت بدون optimizerِ `manifest.json`، `coverage.json`، `metrics_summary.csv` و SVG توسط `finalize_existing.py` انجام می‌شود.
- این checkpoint گواه موفقیت نهایی نیست؛ provenance نهایی مربوط به کامیت اجرای اولیه `66a007928e129ad6dd5acffeaa93fc2485ee9775` می‌ماند. کل commitهای بعدی صرفاً recovery و reporting هستند. هر عدم تطابق یا شکست بازخوانی باید `PILOT_BLOCKED` شود و raw حفظ شود.
- گیت بعدی: `PILOT_READY_FOR_QA` فقط در صورت موفقیت بازیابی و ممیزی مستقلی که مدیر تعیین کند. PR #79 همچنان Draft و unmerged است.

## Recovery attempt 1 (zero training)
[CI #43614895487](https://github.com/mahhammad/My-Thesis/actions/runs/43614895487): دانلود هر ۸ فایل raw موفق شد؛ کنترل fingerprint با پیام کلی FAIL شد و بدون گزارش عددی توقف کرد. کد بازیابی فقط برای چاپ کلید و هش نامنطبق سخت‌گیرانه‌تر شد؛ هیچ تغییری در raw یا optimizer وجود ندارد. اجرای بازیابی بعدی هم فقط read/recompute است.
