# وضعیت باز و نقشه تکمیل پایان‌نامه IMC
**CURRENT / SINGLE BACKLOG**. هر موضوع این صفحه یا *کار لازم برای متن مورد ادعا* است یا *پژوهش اختیاری*. باز بودن تاریخی Issue به معنی اجرای اجباری نیست. تصمیم نهایی ترتیب و پایان پروژه با مالک است.

## کارهای واقعاً باز برای پایان‌نامه
| کار | اولویت/وضعیت | معیار بستن و شاهد | Issue |
|---|---|---|---|
| فصل پنجم و دادهٔ واقعی | **REQUIRED_FOR_THESIS** در برنامهٔ فصل پنجم (تا زمانی که مالک تصمیم رسمی دربارهٔ حذف/کاهش دامنه نگرفته است) | تعیین پرسش/مجموعه‌داده مجاز و split، قانون استنتاج معتبر، CatBoost/RandomForest هم‌بودجه، raw/seed/manifest و ارزیابی Test قفل‌شده؛ گزارش محدودیت و نگارش فصل۵ | **مدیر باید مأموریت جدید بدهد**؛ در Issueهای موجود اجرای فصل۵ قفل/تحویل نشده |
| آزمون بیرونی مستقل بعد از قفل مدل منتخب | **REQUIRED_FOR_THESIS_IF_FINAL_GENERALIZATION_CLAIMED**؛ فعلاً گیت تصمیم/اجرا باز | پیش‌ثبت داده و seed کاملاً تازه، عدم انتخاب model/τ/seed/checkpoint با Test، ارزیابی shared/class (قابل استنتاج) و baseline بیزی؛ تعریف دقیق scope؛ در صورت عدم نیاز به ادعای تعمیم جدید، مالک باید صریحاً از دامنه خارج کند | [#39](https://github.com/mahhammad/My-Thesis/issues/39) — **KEEP_OPEN**؛ پیش‌نیاز تاریخی F در #37 دیگر اجبار نیست |
| تکمیل کیفیت نشر/متن نهایی | **REQUIRED_FOR_THESIS** | مالک نسخه فعلی را با XeLaTeX، شکل‌ها، فهرست نماد/کتاب‌نامه و PDF نهایی کنترل کند؛ بر اساس آخرین PR ادغام‌شده و chapter5؛ هر مسئله مشاهده‌شده با لاگ و صفحه | [#29](https://github.com/mahhammad/My-Thesis/issues/29) برای ادغام فصل۳/۴ تاریخی تکمیل؛ گیت چاپ نهایی فصل۵ مستقل است |
| تصمیم انتشار سنجه‌های غیرمرکزمحور | **REQUIRED_ONLY_IF_CLAIMED**؛ به‌صورت اجراشده فرض نشود | برای هر NLL/BA/حاشیه یا ۴۴ فیلد خارج از QA76 به raw و تست و QA لازم پیوند بده؛ اگر در متن ادعای پذیرفته‌شده جدیدی نیست، نیاز به full rerun نیست | [#38](https://github.com/mahhammad/My-Thesis/issues/38) — تا تعیین تکلیف pipeline و قرارداد **KEEP_OPEN** |
| تأیید بازماندهٔ اختلاف نتیجه/استنتاج تاریخی | **MANAGER_DECISION / NOT_AUTOMATIC_EXECUTION** | بررسی اینکه آیا متن جاری اصلاً به مقایسه mean π یا شواهد قدیمی نیاز دارد؛ در غیر این صورت به IDEA-009 منتقل است | [#33](https://github.com/mahhammad/My-Thesis/issues/33) در ایده ثبت شد |

## کارهای اختیاری؛ پیش‌نیاز پایان‌نامه نیستند
| بسته | طبقه | منبع پژوهش و یادداشت |
|---|---|---|
| مقایسه کنترل‌شده lossها / τ، چندشروعی و ظرفیت Z | OPTIONAL_EXTENSION | [IDEA-006](../research/future-ideas/IDEA-006-objectives-multistart-gradient-conflict.md)، Issues #22/#23/#36 |
| تعارض گرادیان تک‌نمونه و پایداری حاشیه سیاست/مرجع | OPTIONAL_EXTENSION | [IDEA-006](../research/future-ideas/IDEA-006-objectives-multistart-gradient-conflict.md)، #24/#31 |
| نویز/missingness/تغییر K و مولد، مطالعه مقیاس‌پذیری | OPTIONAL_EXTENSION | [IDEA-007](../research/future-ideas/IDEA-007-robustness-generator-structure.md)، #26 |
| بهینه‌سازی روی فیبر هم‌مرکز و اثر بر BA | OPTIONAL_EXTENSION | [IDEA-002](../research/future-ideas/IDEA-002-fixed-center-policy-optimization.md)، #35 |
| فراوانی نقض کران بیزی و قیود مشترک قوی‌تر | OPTIONAL_EXTENSION | [IDEA-003](../research/future-ideas/IDEA-003-bayes-center-reachability-frequency.md)، [IDEA-008](../research/future-ideas/IDEA-008-joint-center-attainability.md)، #32 |
| گراف تنک و هزینه واقعی n بزرگ | OPTIONAL_EXTENSION | [IDEA-004](../research/future-ideas/IDEA-004-sparse-feature-graph.md)، #21 |
| تابع F قابل‌تعمیم برای policy آزاد | OPTIONAL_EXTENSION | [IDEA-005](../research/future-ideas/IDEA-005-functional-policy-F.md)، #37 |
| قواعد استنتاج و میانگین π در مقابل فرض کلاس | OPTIONAL_EXTENSION | [IDEA-009](../research/future-ideas/IDEA-009-inference-protocol-comparisons.md)، #33/#34 |
| پوشش چندکی شعاع KL و پالایش outlier | OPTIONAL_EXTENSION | [IDEA-001](../research/future-ideas/IDEA-001-quantile-kl-radius.md)، طرح از پیش ثبت‌شده و **اجرانشده** |

## کار انجام‌شده و نیازمند جلوگیری از تکرار
- فصل۳ و پیوست/فهرست نمادها با PR #76 در main ادغام؛ مطالعه چرنوف و هندسه JSD دارای برهان در متن. Issue بازِ قدیمی درباره نبود برهان، وضعیت امروز را نشان نمی‌دهد.
- فصل۴ تا §۴.۱۲ با PRهای #61/#62/#65/#66/#71/#72 ادغام شده و گزارش‌ها/بسته‌های شواهد محفوظ‌اند. بررسی دوباره ۱۰۸ run یا ۷۶ مرکز فقط به خاطر قدیمی‌بودن تاریخ Issue **الزام نیست**.
- ۹ ZIP تاریخی آرشیو و manifest خود را دارند. خطاهای MC/نرمال‌سازی/BL V1 در مدارک اصلاحیه‌اند؛ نسخه اول را «پذیرفته‌شده» ننام.
- همه موارد حساس به QA به وضعیت واقعی PASS/FAIL/NOT_RUN متکی‌اند؛ هیچ پژوهش تازه فقط با فهرست قدیمی P00–P16 مجوز اجرا نمی‌گیرد.

## سؤال‌های مدیریتیِ باز
۱. فصل پنجم دقیقاً چه داده‌ها و baselineهایی خواهد داشت؟ ۲. آیا Test کاملاً تازه به‌عنوان شرط دفاع الزامی است یا فقط در صورت ادعای تعمیم جدید؟ ۳. آیا سنجه‌های غیرمرکزمحور §۴.۱۲ برای متن نهایی نیازمند QA تکمیلی‌اند؟ این تصمیم‌ها قبل از تخصیص آزمایش سنگین لازم‌اند.
