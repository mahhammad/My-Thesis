# گزارش ممیزی تجمیع دانش و پاک‌سازی Issueهای پایان‌نامه
**Mission:** `RESEARCH-KNOWLEDGE-CONSOLIDATION-01`، مسئول `RESEARCH-KNOWLEDGE-CURATOR-01`، تاریخ ۲۰۲۶-۱۱-۱۴؛ **مبنای قفل‌شده `main@aa98a5614d28bb8ab51fb2774faf7d6e69b6fa7c`**، [Draft PR #77](https://github.com/mahhammad/My-Thesis/pull/77)، شاخه `editorial/research-knowledge-consolidation-01`. **گیت:** `MANAGER_REVIEW`، ادغام خودکار ممنوع.

## الف) پوشش منابع و روش بررسی
| نوع | شمارش قطعی مشاهده‌شده | نحوه حفاظت/بررسی |
|---|---:|---|
| Research Journal در `main` پیش از کار | **۳۱ فایل** | محتوای اسناد و سرفصل‌ها بررسی و در [PROVENANCE](PROVENANCE_FA.md) برای هر فایل **SHA blob و مقصد** تعیین شد. هیچ ژورنال تاریخ‌دار حذف یا جابه‌جا نشد. |
| Issueهای GitHub | **۳۶ = در آغاز ۳۴ باز + ۲ بسته؛ در پایان ۲ باز + ۳۴ بسته** | متن، وضعیت و عنوان ۳۶ مورد از API زنده خوانده؛ برای هر Issue مقصد و علت رده A/B/C/D/E مشخص شد. |
| کامنت‌های Issue #49 | **۱۱۱ کامنت کامل** | متن‌ها از API خوانده شد؛ [نمایهٔ ۱۱۱ ردیفی](archive/ISSUE49_COMMENT_INDEX_FA.md) و زنجیرهٔ رأی/نقض/تحویل/مرج؛ متن اصلی در GitHub باقی می‌ماند. |
| PRها | **۴۱ ردیف metadata برگشتی در صفحه اول API** (نه ادعای پوشش تمام تاریخچه) | وضعیت merged/closed و SHA مرج، ۲۱ PR کلیدی در [PROVENANCE](PROVENANCE_FA.md) ردیابی شد؛ PR #76 مرج شده؛ PRهای #50/#51/#64/#73 بسته و بی‌مرج هستند. |
| اسناد آزمایش | **۴۵ README/REPORT/RESULTS/CONSOLIDATION** در `code/experiments/` | ۴۵ سند نسخهٔ `main` خوانده و سرفصل/گیت‌ها وارسی شد؛ raw/ZIP/manifest در همان جای خود دست‌نخورده ماندند. نمایه سنجه/نسخه در [EXPERIMENTS](EXPERIMENTS_FA.md). |
| پیش‌نویس‌های R02 خارج از main | **۲ فایل قابل بازیابی** | متن کامل و Git blob اصلی در [archive/unmerged-r02](archive/unmerged-r02/2026-11-14-ch3-bellman-two-sided-proof-and-feasibility-bridge.md) حفظ شد؛ این پیش‌نویس‌ها جایگزین متن پذیرفته‌شدهٔ فصل سوم نیستند. یادداشت تاریخی مثال نقض از SHA بررسی‌شده بازیابی نشد و منبع رأی Issue #49 در نمایه حفظ است. |
| ایده‌های آینده | **۱ فایل اولیه** (IDEA-001) | بدون تغییر حفظ شد؛ IDEA-002 تا 009 هشت فرضیه/پروتکل اختیاری جدید با منشأ Issue ایجاد شدند. |

## ب) انتقال دانش و حفاظت مسیر
| منبع تاریخی | مقصد فعال و نوع انتقال | تضمین/مرز |
|---|---|---|
| نظریه فصل۳ و ۳۱ ژورنال موجود در main + دو پیش‌نویس تاریخی R02 | [THEORY_FA](THEORY_FA.md) + [SCIENTIFIC_DECISIONS_FA](SCIENTIFIC_DECISIONS_FA.md) + [موجودی ۳۱ ردیفی](PROVENANCE_FA.md) | نسخه معتبر فقط در [LaTeX](../tex/chapter3.tex)/پیوست و گزارش نهایی؛ **تمام ۳۰ ژورنال تاریخ‌دار در مسیر قبلی حفظ شدند** و دو سند تاریخی دیگر با blob برابر در آرشیو ثبت شدند. |
| مسائل و لاگ‌های فصل۴ | [EXPERIMENTS_FA](EXPERIMENTS_FA.md)، [پیوست الکترونیکی](../code/experiments/README_FA.md) | raw, SHA/manifest و وضعیت پذیرش محدود محفوظ؛ ۷۶ run ≠ ۷۶ seed؛ ۸۳۶ center-only ≠ تمام سنجه‌ها. |
| Issue #49؛ ۱۱۱ کامنت تصمیم و گیت | [archive/ISSUE49_COMMENT_INDEX_FA.md](archive/ISSUE49_COMMENT_INDEX_FA.md) + SCIENTIFIC_DECISIONS/EXPERIMENTS | هیچ کامنت پاک نشد. نقض نظری، numeric V1 fail، V2 scoped PASS، B02 verifier gate و انتشار #61/#62/#65/#66/#72 جدا شده‌اند. |
| Issueهای #52–#60 | [WORKFLOW_FA](WORKFLOW_FA.md) + [PROVENANCE_FA](PROVENANCE_FA.md) | پروتکل دائمی شدن شاخه researcher و Issue-only workflow **SUPERSEDED**؛ تاریخچه قابل دسترس است. |
| index قدیمی `research_journal/README_FA.md` | [archive/README_FA-legacy-2026-11-14.md](archive/README_FA-legacy-2026-11-14.md) | متن نسخه اصلی بدون حذف آرشیو، README فعال فقط ناوبری جاری. |
| طرح‌های کار اختیاری P00–P16 | [OPEN_RESEARCH_FA](OPEN_RESEARCH_FA.md)، [IDEA-001…009](../research/future-ideas/) | تکلیف علمی، شواهد پذیرش و فرضیه آینده تفکیک؛ هیچ پژوهش اجرا نشده را انجام‌شده ننامیده‌ایم. |

**تصمیم عدم حذف فیزیکی ژورنال‌های قدیمی:** به‌سبب ارجاع ثابت تعداد زیادی از Issues، گزارش‌ها و نام‌های commit به مسیرهای قدیمی، جابه‌جایی گروهی سبب شکستن لینک می‌شد. این اسناد به‌عنوان historical evidence در جای خود باقی و تنها پنج–هفت سند موضوعی در README به‌صورت فعال توصیه شده‌اند. Git history به تنهایی تکیه‌گاه بازیابی نیست.

## ج) یکپارچه‌سازی دستورالعمل‌های قدیمی
| مرجع تاریخی | تعارض | جایگزین معتبر در این PR |
|---|---|---|
| Issue #55 | «دستور فقط در Issue؛ دانش در branch» در برابر تقاضای کاربر برای سند واحد در main | [WORKFLOW_FA.md](WORKFLOW_FA.md): قرارداد عمومی واحد؛ Issue صرفاً مأموریت و رأی محدود |
| #52/#53 | جانشینی، snapshotهای تاریخ‌دار و اولویت متغیر مدیر | WORKFLOW بخش انتقال؛ وضعیت جاری OPEN_RESEARCH و PROVENANCE |
| #56/#57 | چند شاخه دائمی، سقف نه‌گانه، برنامه پاک‌سازی با SHA تاریخی | فقط main دائمی و یک شاخه موقت/PR؛ حذف پس از ادغام و تأیید مالک |
| #58/#59/#60 | دفترهای «دائمی» پژوهشگران همراه شاخه‌های دائمی | گیت‌ها/شواهد به تصمیم‌های علمی، آزمایش‌ها و فهرست منشأ منتقل شد |
| #49 | دستورها و طرح‌های وابسته به دوره‌ای که §§۴.۱۰–۴.۱۲ هنوز منتشر نشده بودند | زنجیره تاریخی در index ۱۱۱ کامنت؛ کار جاری در OPEN_RESEARCH |
| ژورنال‌های نقشه راه ۹–۱۱ نوامبر | آزمایش‌های برنامه‌ریزی شده ممکن است بعداً انجام شده باشند | هر کار از فایل raw/manifest و گزارش نهایی ارزیابی شد، نه از checkbox تاریخی |

## د) جدول تک‌به‌تک تصمیم Issueها
**وضعیت نهایی تأییدشده با بازخوانی GitHub:** ۳۶ Issue، ۳۴ بسته و دو Issue باز (#38 و #39). از این شمار، ۳۲ Issue در همین مأموریت پس از ثبت کامنت مستند بسته شدند؛ #25 و #27 از ابتدا بسته بودند. هر مورد بسته‌شده لینک مقصد و PR #77 دارد. موارد #38/#39 برای تصمیم مدیریتی/اجرایی حفظ شدند. دسته‌ها: A تکمیل؛ B superseded؛ C اجرایی باز؛ D پژوهش آینده؛ E مبهم/نیازمند تصمیم.

| Issue | رده | مقصد انتقال | وضعیت عملیات |
|---|---|---|---|
| [#21](https://github.com/mahhammad/My-Thesis/issues/21) | D | IDEA-004 | **بسته شد + کامنت انتقال** |
| [#22](https://github.com/mahhammad/My-Thesis/issues/22) | D | IDEA-006 | **بسته شد + کامنت انتقال** |
| [#23](https://github.com/mahhammad/My-Thesis/issues/23) | D | IDEA-006 | **بسته شد + کامنت انتقال** |
| [#24](https://github.com/mahhammad/My-Thesis/issues/24) | D | IDEA-006 | **بسته شد + کامنت انتقال** |
| [#25](https://github.com/mahhammad/My-Thesis/issues/25) | B | مأموریت قبلاً بسته بود | بسته از قبل |
| [#26](https://github.com/mahhammad/My-Thesis/issues/26) | D | IDEA-007 | **بسته شد + کامنت انتقال** |
| [#27](https://github.com/mahhammad/My-Thesis/issues/27) | A | آرشیو ۹ ZIP قبلاً بسته بود | بسته از قبل |
| [#28](https://github.com/mahhammad/My-Thesis/issues/28) | A | EXPERIMENTS_FA + PR #72 | **بسته شد + کامنت انتقال** |
| [#29](https://github.com/mahhammad/My-Thesis/issues/29) | A | PR #76؛ گیت چاپ فصل۵ جداست | **بسته شد + کامنت انتقال** |
| [#30](https://github.com/mahhammad/My-Thesis/issues/30) | A | THEORY_FA؛ کران چرنوف متن نهایی | **بسته شد + کامنت انتقال** |
| [#31](https://github.com/mahhammad/My-Thesis/issues/31) | D | IDEA-006 | **بسته شد + کامنت انتقال** |
| [#32](https://github.com/mahhammad/My-Thesis/issues/32) | D | IDEA-008 + IDEA-003 | **بسته شد + کامنت انتقال** |
| [#33](https://github.com/mahhammad/My-Thesis/issues/33) | D | IDEA-009 | **بسته شد + کامنت انتقال** |
| [#34](https://github.com/mahhammad/My-Thesis/issues/34) | D | IDEA-005 + IDEA-009 | **بسته شد + کامنت انتقال** |
| [#35](https://github.com/mahhammad/My-Thesis/issues/35) | D | IDEA-002 | **بسته شد + کامنت انتقال** |
| [#36](https://github.com/mahhammad/My-Thesis/issues/36) | D | IDEA-005 | **بسته شد + کامنت انتقال** |
| [#37](https://github.com/mahhammad/My-Thesis/issues/37) | D | IDEA-005 | **بسته شد + کامنت انتقال** |
| [#38](https://github.com/mahhammad/My-Thesis/issues/38) | E | OPEN_RESEARCH؛ تصمیم گیت معیار و snapshot | **OPEN / با معیار پذیرش جدید** |
| [#39](https://github.com/mahhammad/My-Thesis/issues/39) | C | OPEN_RESEARCH؛ Test مستقل پس از قفل مدل | **OPEN / با معیار پذیرش جدید** |
| [#40](https://github.com/mahhammad/My-Thesis/issues/40) | A | تحویل E08 و PR #42/#43 | **بسته شد + کامنت انتقال** |
| [#41](https://github.com/mahhammad/My-Thesis/issues/41) | A | روایت N-CH4-01 و PR #43/#48 | **بسته شد + کامنت انتقال** |
| [#44](https://github.com/mahhammad/My-Thesis/issues/44) | A | گزارش H و PR #46 | **بسته شد + کامنت انتقال** |
| [#45](https://github.com/mahhammad/My-Thesis/issues/45) | A | گزارش H و PR #46 | **بسته شد + کامنت انتقال** |
| [#47](https://github.com/mahhammad/My-Thesis/issues/47) | A | THEORY_FA؛ سازگاری gτ در PR #76 | **بسته شد + کامنت انتقال** |
| [#49](https://github.com/mahhammad/My-Thesis/issues/49) | B | ISSUE49_COMMENT_INDEX_FA (۱۱۱ کامنت) + تصمیم‌ها | **بسته شد + کامنت انتقال** |
| [#52](https://github.com/mahhammad/My-Thesis/issues/52) | B | WORKFLOW_FA | **بسته شد + کامنت انتقال** |
| [#53](https://github.com/mahhammad/My-Thesis/issues/53) | B | WORKFLOW_FA + OPEN_RESEARCH_FA | **بسته شد + کامنت انتقال** |
| [#54](https://github.com/mahhammad/My-Thesis/issues/54) | B | README_FA + PROVENANCE_FA | **بسته شد + کامنت انتقال** |
| [#55](https://github.com/mahhammad/My-Thesis/issues/55) | B | WORKFLOW_FA (تنها مرجع اجرایی) | **بسته شد + کامنت انتقال** |
| [#56](https://github.com/mahhammad/My-Thesis/issues/56) | A | PROVENANCE_FA؛ ممیزی قبلی کامل، حذف با مالک | **بسته شد + کامنت انتقال** |
| [#57](https://github.com/mahhammad/My-Thesis/issues/57) | B | WORKFLOW_FA؛ برنامه سقف ۹ شاخه منسوخ | **بسته شد + کامنت انتقال** |
| [#58](https://github.com/mahhammad/My-Thesis/issues/58) | B | SCIENTIFIC_DECISIONS_FA؛ نقش پژوهشگر ۲ | **بسته شد + کامنت انتقال** |
| [#59](https://github.com/mahhammad/My-Thesis/issues/59) | B | EXPERIMENTS_FA؛ نقش پژوهشگر ۳ | **بسته شد + کامنت انتقال** |
| [#60](https://github.com/mahhammad/My-Thesis/issues/60) | B | EXPERIMENTS_FA؛ نقش پژوهشگر ۵ | **بسته شد + کامنت انتقال** |
| [#67](https://github.com/mahhammad/My-Thesis/issues/67) | A | PR #68/#69/#76 | **بسته شد + کامنت انتقال** |
| [#70](https://github.com/mahhammad/My-Thesis/issues/70) | A | PR #76 + THEORY_FA | **بسته شد + کامنت انتقال** |

## هـ) ایده‌ها و پرسش‌های آینده
| شناسه | انگیزه و وابستگی | گیت |
|---|---|---|
| [IDEA-001](../research/future-ideas/IDEA-001-quantile-kl-radius.md) | شعاع KL و پوشش چندکی؛ کنترل outlier/reweight؛ فرضیهٔ بهبود هنوز اثبات نشده | OPTIONAL_EXTENSION |
| [IDEA-002](../research/future-ideas/IDEA-002-fixed-center-policy-optimization.md) | تغییر زیان در فیبر هم‌مرکز، تصحیح دقیق مرکز و BA | OPTIONAL_EXTENSION |
| [IDEA-003](../research/future-ideas/IDEA-003-bayes-center-reachability-frequency.md) | فراوانی نقض شرط لازم بلمن در ساختارهای داده | OPTIONAL_EXTENSION |
| [IDEA-004](../research/future-ideas/IDEA-004-sparse-feature-graph.md) | کاهش هزینه گراف تنک با شرط حفظ مقصد مجاز | OPTIONAL_EXTENSION |
| [IDEA-005](../research/future-ideas/IDEA-005-functional-policy-F.md) | F قابل تعمیم برای سیاست آزاد node/node_class | OPTIONAL_EXTENSION |
| [IDEA-006](../research/future-ideas/IDEA-006-objectives-multistart-gradient-conflict.md) | زیان‌های متقابل، چندشروعی و conflict gradient | OPTIONAL_EXTENSION |
| [IDEA-007](../research/future-ideas/IDEA-007-robustness-generator-structure.md) | حساسیت نویز/mask/K/مولد جدید | OPTIONAL_EXTENSION |
| [IDEA-008](../research/future-ideas/IDEA-008-joint-center-attainability.md) | قید مشترک فراتر از جعبه بلمن | OPTIONAL_EXTENSION |
| [IDEA-009](../research/future-ideas/IDEA-009-inference-protocol-comparisons.md) | سه قرارداد استنتاج و میانگین π | OPTIONAL_EXTENSION |

## و) فقط کارهای واقعاً ماندهٔ پایان‌نامه
| کار | وضعیت | تصمیم لازم |
|---|---|---|
| **فصل پنجم / دادهٔ واقعی و baselineهای RF/CatBoost** | `REQUIRED_FOR_THESIS` در برنامهٔ فعلی | تعیین دامنه/داده/بودجه و مأموریت مستقل؛ هیچ اجرا/نتیجه جدیدی در این PR ایجاد نشده |
| **Test کاملاً مستقل پس از قفل مدل** | `REQUIRED_IF_FINAL_GENERALIZATION_CLAIMED`، [Issue #39](https://github.com/mahhammad/My-Thesis/issues/39) باز | مالک تعیین کند آیا به گیت نهایی اضافه شود؛ پیش‌نیاز F اختیاری را تحمیل نکن |
| **گیت انتشار سنجه‌های بیشتر از QA76** | `REQUIRED_ONLY_IF_CLAIMED`، [Issue #38](https://github.com/mahhammad/My-Thesis/issues/38) باز | ابتدا مطالعه کن آیا متن نهایی واقعاً به ادعای افزوده نیاز دارد؛ قبل از آن rerun ممنوع |
| **ساخت و کنترل نهایی PDF بعد از فصل۵** | `REQUIRED_FOR_THESIS` | مالک با XeLaTeX/فهرست/منابع/تصاویر؛ این مستندسازی نیاز به build جدید ندارد |
| زیان، چندشروعی، conflict gradient، نویز، K، مقیاس، F، مسائل فیبر | `OPTIONAL_EXTENSION` | پرونده IDEA، نه الزام ارشد |
| کران بلمن، ۱۰۳ واحد §۴.۱۱، ۱۰۸ §۴.۱۰، ۷۶/۸۳۶ §۴.۱۲، انتقال فصل۳/۴ | `COMPLETED_OR_SUPERSEDED` با محدوده رأی | تکرار خودکار ممنوع؛ داده خام ثابت |

## ز) Git، کنترل‌های پذیرش و مرز ادعا
- **ساخت شاخه:** فقط `editorial/research-knowledge-consolidation-01` از SHA مرج PR #76. **PR:** #77 Draft به main؛ بدون مرج.
- **عملیات Issue:** ۳۲ مورد با کامنت نهایی و برچسب `completed` یا `not_planned` بسته شدند؛ #25/#27 از قبل بسته بودند. #38/#39 با عنوان و شرح فعلی و معیار پذیرش باز مانده‌اند. **open=2, closed=34**.
- **فایل‌های تغییر‌یافته:** فقط مستندات `research_journal/` و `research/future-ideas/`؛ [مقایسه PR](https://github.com/mahhammad/My-Thesis/pull/77/files). هیچ `tex/`، `code/`، `results/`، zip یا dataset تغییر نکرد.
- **شاخهٔ پیشین:** `editorial/ch3-language-reference-audit-01` ابتدا بعد از مرج وجود داشت؛ مالک حذف آن را پس از اجازه مدیر اعلام کرد و نبودن ref با بازخوانی تأیید شد؛ curator هیچ شاخه‌ای حذف نکرد.
- **کنترل‌ها:** **PASS (STATIC)** — بررسی تطبیقی diff با `main` نشان داد تمام تغییرات فقط در اسناد `research_journal/` و `research/future-ideas/` است. وجود فایل مقصد تمام پیوندهای نسبی در ۲۱ سند موضوعی/ایده/آرشیو بررسی شد و پس از بازپیونددهی ۲۵ نشانی نسبی نسخهٔ تاریخی README، **هیچ مقصد نسبی مفقود شناسایی نشد**. تعداد ۳۱ ژورنال اولیه و ۳۶ Issue (۳۲ بستن جدید + دو ازقبل‌بسته + دو باز)، ۱۱۱ کامنت #49، هشت IDEA جدید و IDEA-001 محفوظ بررسی شد. دو نسخهٔ منحصراً تاریخی R02 با Git blob **عیناً برابر** آرشیو شدند. [README تاریخی](archive/README_FA-legacy-2026-11-14.md) فقط در نشانی‌های نسبی و یادداشت وضعیت بازپیونددهی شده؛ نسخهٔ اولیه با blob `14ba708cf1195b752f1054fef4cfd14264ebcea7` در مرج مبنا محفوظ است. آزمون کد/XeLaTeX/PDF `NOT_RUN / OUT_OF_SCOPE` است؛ صحت خودکار تمام URLهای بیرونی آزمایش نشد.
- **گیت نهایی:** **`READY_FOR_MANAGER_KNOWLEDGE_REVIEW`**؛ تمامی عملیات مستند انجام شد و فقط بررسی و تصمیم مدیر دربارهٔ Draft PR #77 باقی است. **هیچ ادغام یا حذف شاخه مأموریت انجام نشده است**.
