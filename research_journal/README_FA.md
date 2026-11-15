# راهنمای واحد مطالعه و نگهداری دانش پایان‌نامه IMC
**CURRENT / START HERE — ۱۴ نوامبر ۲۰۲۶.** این README ورودی **مرجع پژوهشی جاری** است؛ اسناد تاریخ‌دار و Issues قدیمی خودشان مرجع دستورالعمل زنده نیستند.

## ترتیب مطالعهٔ سریع برای پژوهشگر و مدیر جدید
1. [**WORKFLOW_FA.md**](WORKFLOW_FA.md) — **تنها قرارداد معتبر عمومی** Git، مأموریت، آزمایش، ممیزی، CI/PDF، جانشینی و حذف شاخه. شاخهٔ دائمی فقط main است.
2. [**THEORY_FA.md**](THEORY_FA.md) — روش IMC، فرض‌ها، فرمول‌ها، قضیه بلمن و محدوده‌های اثبات؛ برای اثبات کامل به [فصل سوم](../tex/chapter3.tex) و [پیوست](../tex/appendix_chapter3.tex) رجوع کن.
3. [**EXPERIMENTS_FA.md**](EXPERIMENTS_FA.md) — نمایهٔ شواهد پذیرفته‌شده و دادهٔ خام و گزارش‌های §§۴.۳–۴.۱۳، همراه طرح §۴.۱۴؛ [متن فصل چهارم](../tex/chapter4.tex)، [پیوست الکترونیکی](../code/experiments/README_FA.md).
4. [**SCIENTIFIC_DECISIONS_FA.md**](SCIENTIFIC_DECISIONS_FA.md) — نسخه صحیح هر ادعا، گیت ممیزی، نتایج منسوخ و حد پذیرش.
5. [**OPEN_RESEARCH_FA.md**](OPEN_RESEARCH_FA.md) — **تنها بک‌لاگ جاری** برای کارهای ضروری پایان‌نامه و توسعه اختیاری؛ Issueهای قدیمی P00–P16 ملاک اولویت امروز نیستند.
6. [**PROVENANCE_FA.md**](PROVENANCE_FA.md) — جدول نسخه‌دار **۳۱ ژورنال، ۳۶ Issue و PRهای اثرگذار**، SHA/منشأ و مقصد انتقال.
7. [**KNOWLEDGE_AUDIT_2026-11-14_FA.md**](KNOWLEDGE_AUDIT_2026-11-14_FA.md) — گزارش ممیزی این تجمیع، حسابرسی عملیات Git و Issueها.
8. [**archive/**](archive/) — [نمایه قدیمی (کپی سالم)](archive/README_FA-legacy-2026-11-14.md)، [۱۱۱ کامنت پیش از اختتام Issue مادر #49](archive/ISSUE49_COMMENT_INDEX_FA.md) و جدول لینک به ژورنال‌های تاریخ‌دار. نسخه‌های قبلی در مسیرهای تاریخی **حذف نشده‌اند**.

### بایگانی دو پیش‌نویس نظری خارج از main

دو پیش‌نویس R02 که فقط در کامیت‌های تاریخی وجود داشتند، **با متن و Git blob اصلی** در [آرشیو برهان بلمن](archive/unmerged-r02/2026-11-14-ch3-bellman-two-sided-proof-and-feasibility-bridge.md) و [تعبیر پسین بیزی](archive/unmerged-r02/2026-11-14-sec411-bayes-posterior-vs-neighborhood-reachability.md) نگهداری شدند. همچنین [ژورنال اولیه مثال نقض هستهٔ بازگشت‌مجاز D10/D11](archive/unmerged-r02/2026-11-11-d10d11-return-kernel-counterexample.md) که در تلاش اولیه قابل بازیابی نبود از commit تاریخی `6bff5773` بازیابی و با blob یکسان آرشیو شد. این **سه سند تاریخی** مرجع وضعیت فعلی قضایا نیستند و باید همراه [تصمیم‌های علمی معتبر](SCIENTIFIC_DECISIONS_FA.md) خوانده شوند. [جزئیات منشأ](PROVENANCE_FA.md).

## ایده‌های آینده — محل ثبت یکتا
[IDEA-001: منحنی پوشش چندکی شعاع KL](../research/future-ideas/IDEA-001-quantile-kl-radius.md)؛
[IDEA-002: فیبر هم‌مرکز](../research/future-ideas/IDEA-002-fixed-center-policy-optimization.md)؛
[IDEA-003: فراوانی نقض بلمن](../research/future-ideas/IDEA-003-bayes-center-reachability-frequency.md)؛
[IDEA-004: گراف تنک](../research/future-ideas/IDEA-004-sparse-feature-graph.md)؛
[IDEA-005: سیاست تابعی F](../research/future-ideas/IDEA-005-functional-policy-F.md)؛
[IDEA-006: زیان/چندشروعی/گرادیان](../research/future-ideas/IDEA-006-objectives-multistart-gradient-conflict.md)؛
[IDEA-007: نویز/mask/K](../research/future-ideas/IDEA-007-robustness-generator-structure.md)؛
[IDEA-008: قیود مشترک](../research/future-ideas/IDEA-008-joint-center-attainability.md)؛
[IDEA-009: استنتاج و میانگین احتمال](../research/future-ideas/IDEA-009-inference-protocol-comparisons.md).

**ایده آینده ≠ نقص اثبات و ≠ مأموریت مجاز برای اجرای آزمایش.** IDEA-001 دارای سابقه و تعریف پیشین است و عیناً حفظ شده؛ ۸ ایده دیگر در مأموریت فعلی بر اساس Issues و منابع واقعی ساخته شده‌اند.

## وضعیت شناخته‌شده پژوهش
- PR #76 در main با SHA **aa98a5614d28bb8ab51fb2774faf7d6e69b6fa7c** مرج شده؛ فصل سوم از لحاظ نظری با فصل چهارم تطبیق یافت. برای ممیزی دقیق [گزارش نهایی](2026-11-14-ch3-full-scientific-editorial-audit.md) (۶۲ ادعا: ۵۹ A/سه B/صفر C/D و دو R) را بخوان.
- شواهد §۴.۱۰ دارای ۱۰۸ ردیف از **سه مولد مستقل** است؛ §۴.۱۱ دارای گواه محدود شش گراف و قضیه بلمن لازم عمومی؛ §۴.۱۲ دارای ۷۶ مرکز و **۸۳۶ سنجه center-only** پذیرفته است، نه همهٔ سنجه‌ها.
- دستورهای Issue #49 و #52–#60 **تاریخی و فاقد اولویت اجرایی جاری**‌اند؛ متون کامل حفظ شده‌اند و تصمیم‌های معتبر در WORKFLOW و SCIENTIFIC_DECISIONS نقل یا لینک شده‌اند.
- نقل عدد یا SHA از جدول‌های ناوبری به‌جای raw/manifest پذیرفته نیست. نتیجه تازه باید آزمایش واقعی، گیت و metadata مستقل داشته باشد.

## نحوه به‌روز‌رسانی
پس از هر مأموریت و مرج، اسناد موضوعی و PROVENANCE را با **رأی محدود به دامنه، SHA ثابت و لینک فایل اصلی** به‌روز کن؛ نسخه تاریخی را تخریب نکن. اگر سند قدیمی باید فیزیکی منتقل شود، اول همه پیوندهای ورودی را بررسی کن؛ تاریخچه Git به‌تنهایی برای بازیابی سریع کافی نیست. **همیشه خواندن وضعیت زنده مخزن و تصمیم جدید مالک بر تاریخ این snapshot مقدم است.**
