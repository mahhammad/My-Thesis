# E-CH4-13 — Checkpoint قفل داده مرحله دوم

**مرحله:** `PHASE2_DESIGN_DATA_FREEZE`؛ پیش از هر fit و بدون Test.
**فرمان جدید مدیر:** ممیزی علمی نهایی پس از مطالعه چندمولدی؛ اجرای مدل صرفاً بعد از ممیزی فنی مقدماتی و مجوز رسمی جداگانه در GitHub. رأی Issue #80 تاکنون ثبت نشده؛ مجوز آموزش همچنان **NO_FIT** است.

**اقدام داده‌ای مجاز و غیرآموزشی:** workflow `.github/workflows/e-ch4-13-phase2-freeze-inputs.yml` سه مجموعه `Train=300,Validation=90` را از seedهای `20261102/03/04` تحت config main و محیط NumPy 2.4.6/OpenBLAS generic یک‌بار می‌سازد؛ هیچ Test و هیچ optimizer وجود ندارد. فایل‌های `inputs/phase2_train_validation/ds<seed>_train_val.npz` و manifest شامل SHA256 bytes/arrays/masks و config تولید و بعد با خوانش مستقل اعتبارسنجی خواهند شد. تغییر یا overwrite ممنوع است.

**هدف علمی:** جلوگیری از تغییر بی‌صدای نمونه‌های Training/Validation در اثر مسیر محاسباتی BLAS بین runnerهای GitHub و حفظ جفتی بودن داده‌های هر seed×family×objective. این داده‌های تازه باید برای تفسیر مقایسه با اجرای تاریخی با fingerprint ارزیابی شوند؛ هم‌نامی seed به‌تنهایی reuse را توجیه نمی‌کند.

**گیت:** صرفاً بعد از اجرای موفق workflow و بررسی مستقل checksum `DATA_FROZEN_NO_FIT` ثبت خواهد شد. این عمل هیچ مجوزی برای ۱۲ آموزش صادر نمی‌کند؛ `phase2_execution_gate.json` باید تا مجوز رسمی مدیر BLOCKED باقی بماند.

**Checkpoint commit trigger:** فقط درخواست قفل ورودی، نه درخواست آموزش.
