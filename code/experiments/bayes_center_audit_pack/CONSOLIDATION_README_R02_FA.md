# CH4-R02-CONSOLIDATE-01 — پژوهشگر ۲، نمایه تجمیع D10/D11

**شاخه پایدار:** `researcher/02-chapter4` · **وضعیت:** `READY_FOR_INDEPENDENT_TREE_QA / OLD_BRANCHES_NOT_DELETED` · **حکم علمی:** `SCIENTIFIC_HOLD`.

این تجمیع یک **merge commit چندوالدی** است؛ parent نخست `recalc-preflight`، دوم `evidence-pack` و سوم ژورنال مثال نقض. تاریخچه `bound-repair → numerical-repair-v2 → recalc-preflight` در parent نخست قرار دارد. بنابراین هر پنج HEAD و tree قدیمی با ancestor قابل بازیابی‌اند، ولی تا تأیید آرشیو و ممیزی **هیچ‌کدام حذف‌شدنی نیستند**. داده، آزمایش، حلگر، متن LaTeX، PRهای ۵۰/۵۱ و `chapter4` دست‌نخورده‌اند.

## جدول منشأ و بقا

| source | commit HEAD | tree SHA | جایگاه / وضعیت |
|---|---|---|---|
| `research/ch4-d10d11-bound-repair` | `75ae4aa54ec412bfbf2d4dd2f531f899f3597145` | `641f5e5e6a3f42770d085c8add418056feac7e7d` | `ANCESTOR_OF_PREFLIGHT` ؛ `HISTORICAL_PROTOTYPE` |
| `research/ch4-d10d11-evidence-pack` | `0cddf6a5995907bc735698874f04335c01d94704` | `a6036f7e2b5e162fed088d956665c61bc644f19b` | `DIRECT_MERGE_PARENT` ؛ `HISTORICAL_INVALID_CERTIFICATE_AND_DESCRIPTIVE_EVIDENCE` |
| `research/ch4-d10d11-numerical-repair-v2` | `9a19dcb2c14f831374eadcb638fa3974e56560e9` | `e8d7a3c8a7e702dae6f2b427aa689f78ea76c9ee` | `ANCESTOR_OF_PREFLIGHT` ؛ `NUMERIC_V2_TOY_ACCEPTED_ONLY` |
| `research/ch4-d10d11-recalc-preflight` | `d87ae396ccad473c436047e36e7766c75f00618d` | `39445944448ae1dbdd5235883af492075a306015` | `FIRST_MERGE_PARENT_AND_BASE_TREE` ؛ `PLAN_ONLY` |
| `research/ch4-bellman-bound-audit-journal-20261111` | `6bff5773aadbd165b65a67ff7430a6c700800fee` | `5b47083b48e4a274ea3b986d257bf217d51f29ae` | `DIRECT_MERGE_PARENT` ؛ `NEGATIVE_AUDIT_OF_HISTORICAL_INVALID_KERNEL` |

### تفکیک اعتبار و خطر تعارض
- **`HISTORICAL_INVALID`**: گواه‌های D10 از محاسبه `full300_return_allowed` (به‌ویژه `H>1`)، نتیجه «عدم امکان equal» و کران پایین تاریخی JSD در D11 را **نباید معتبر بازنشر کرد**. پرونده‌های `results/` و `EDITING_HANDOFF_FA.md` از شاخه evidence دقیقاً با SHA blob اصلی محفوظ‌اند، اما به دلیل ابطال مبنای هسته، برای ادعای قطعی تنها آرشیو/منشأ هستند. حسابداری ۱۸ مرکز و معیارهای تاریخی می‌تواند به‌صورت توصیفی و با همان provenance مطالعه شود؛ نه به‌عنوان گواه عدم دست‌یابی.
- **`NUMERIC_V2_TOY_ACCEPTED_ONLY`**: کد فعال `bound_repair_candidate/exact_origin_excluded_bound.py` = blob `7fe3ae82f10b3120054854f82ee3916c55605f30` از V2؛ پذیرش مستقل نمونه‌های کوچک به معنی اثبات صحت عددی `n=300` یا اجازه اجرای شبکه ۳۷تنظیمی نیست.
- **نسخه تاریخی V1:** کد همان مسیر در `75ae4aa54ec412bfbf2d4dd2f531f899f3597145`، blob `c0b26a8d42fe6985a1b7fcd1c4e3d036f072c68e`. در TIP جایگزین V2 نشده است؛ از تاریخچه commit والد قابل بازیابی است. اختلاف نسخه با `PRESERVED_WITH_CHANGES_ARCHIVED_IN_GIT_ANCESTRY` ثبت می‌شود، نه ادعای برابری blob.
- **`PLAN_ONLY`**: سه سند `recalc_preflight/` از HEAD پیش‌پرواز بدون تغییر است؛ هیچ اجرای علمی از این اسناد استنتاج نمی‌شود.
- **ژورنال مثال نقض:** `research_journal/2026-11-11-d10d11-return-kernel-counterexample.md`، blob `5d47c6229f819484dda22a5012feedc7ef4d3a57`، دقیقاً از شاخه موضوعی حفظ شده است؛ این یک یافته منفی درباره کران قدیم است، نه حکم ناممکن بودن هدف‌ها. متن ژورنال شامل دستور اجرایی تازه نمی‌شود.
- **تعارض حل‌نشده علمی:** اعتبار عددی کران جدید در گراف ۳۰۰تایی، وضعیت نهایی `equal/bounded_equal` و معنای گواه‌های subset72 هنوز به QA و مجوز بازحساب وابسته‌اند. **تعارض فایل هم‌مسیر در انتقال ۹ blob پیدا نشد**.

## ممیزی مستقل و بازنشستگی
از مدیر/ممیز درخواست می‌شود:
۱) پنج `source_head` را با `git merge-base --is-ancestor <source> researcher/02-chapter4` تأیید کند؛
۲) هر ۸ blob evidence و blob ژورنال را در TIP byte-identical بیابد؛
۳) مسیر V2 و سه فایل پیش‌پرواز را با HEAD قفل‌شده مقایسه کند؛
۴) اختلاف تاریخی V1/V2 را از دو commit بازخوانی کند؛
۵) نبود تغییر در `chapter4`، آرشیو raw و PRهای نگارشی را کنترل کند.

**`READY_TO_RETIRE_AFTER_ARCHIVE_AND_QA`** صرفاً فهرست پنج منبع بالا به‌عنوان *کاندید مشروط* است. **`KEEP_UNTIL_GATE`** برای **همه پنج شاخه** تا `ARCHIVE_VERIFIED + INDEPENDENT_TREE_QA + OWNER_APPROVAL` برقرار است. `DELETE_SOURCE_BRANCH_AFTER_MERGE` بر این شاخه‌های تاریخی اعمال نمی‌شود. **هیچ شاخه‌ای حذف نشده است.**

فهرست کامل `source → path → blob_sha → preservation`: [`CONSOLIDATION_MANIFEST_R02.json`](./CONSOLIDATION_MANIFEST_R02.json). دفتر عملیات [Issue #57](https://github.com/mahhammad/My-Thesis/issues/57) و نمایه ژورنال [Issue #54](https://github.com/mahhammad/My-Thesis/issues/54) هستند؛ دستور مدیریتی جدید در ژورنال درج نشده است.
