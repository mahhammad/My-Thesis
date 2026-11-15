# تصمیم‌های علمی جاری و مرز پذیرش
**CURRENT / SCIENCE**. در این جدول وضعیت «اثبات ریاضی»، «پذیرش ممیزی»، «ارزیابی عددی» و «برنامه پژوهشی» یکی نیستند. برای اعداد به [EXPERIMENTS_FA.md](EXPERIMENTS_FA.md) رجوع کنید.

| موضوع | نتیجهٔ معتبر و فرض‌ها | گیت/محدودیت و منبع |
|---|---|---|
| گشت بدون بازگشت | هسته مبدأحذف در همه گام‌های قطعه؛ انتقال نخست جداست | [اثبات فصل۳](../tex/appendix_chapter3.tex)، [اصلاح نظری](2026-11-14-ch3-postch4-theory-alignment.md) |
| Bellman قدیمی | کران‌های با هسته بازگشت‌مجاز یا محاسبهٔ تفریقی ناپایدار **SUPERSEDED** هستند | [نقد نظری #49](https://github.com/mahhammad/My-Thesis/issues/49#issuecomment-6794025754)، [مثال نقض کد V1](https://github.com/mahhammad/My-Thesis/issues/49#issuecomment-6796470635)؛ [ژورنال اصلی مثال نقض، بازیابی با blob برابر](archive/unmerged-r02/2026-11-11-d10d11-return-kernel-counterexample.md) |
| Bellman نهایی | شرط لازم دوطرفه برای *هر مسئله واجد فرض* اثبات شده؛ بیرون بازه بودن معتبر ⇒ هدف ناممکن؛ داخل بودن کافی نیست | [فصل۳/پیوست](../tex/appendix_chapter3.tex)؛ قانون ریاضی به شش نمونه خاص محدود نیست |
| اصلاح عددی V2 | تفریق ناپایدار اصلاح و fixture کوچک مستقل گذرانده شد | **ACCEPT_NUMERIC_V2_FOR_BOUNDED_NEXT_GATE** روی n≤6، **نه** گواه بی‌حد شناور؛ [رأی #49](https://github.com/mahhammad/My-Thesis/issues/49#issuecomment-6797582555)، SHA کاندید 9a19dcb2c14f831374eadcb638fa3974e56560e9 |
| §۴.۱۱ مراکز بیزی | پسین برچسبِ خود نمونه با برچسب مقصد گشت فرق دارد؛ شش گراف پایه دارای گواه عددی در محدوده مشخص هستند | [گزارش رسمی](../code/experiments/bayes_center_reachability_standard/REPORT_FA.md)؛ حساسیت‌های دیگر blanket-certified نیستند |
| B01-v2 | ۱۸ اجرای زوجی و آزمون توقفِ اصلاح‌شده | **ACCEPT_B01_V2** [رأی مستقل](https://github.com/mahhammad/My-Thesis/issues/49#issuecomment-6791688515)؛ این رأی درباره ۱۰۸ اجرای بعدی به همان وسعت نیست |
| §۴.۱۰ — ۱۰۸ مورد | ۱۰۸ رکورد از سه مولد؛ هدف‌های قابل‌تحقق با شاهد سیاست | [گزارش](../code/experiments/inverse_attainable_standard/REPORT_FA.md)؛ audit baseline **ACCEPT_WITH_FIXES**، برخی rawها **PROVISIONAL_PENDING_FINAL_QA**، بازپخش گراف مستقل کامل انجام نشده |
| QA76 هندسه مراکز | ۷۶/۷۶ مرکز و ۸۳۶/۸۳۶ مقدار **center-only** | **ACCEPT_QA76_CENTER_ONLY** [ممیزی #49](https://github.com/mahhammad/My-Thesis/issues/49#issuecomment-6796097800)، [قرارداد §۴.۱۲](../code/experiments/standard_policy_comparison/r03_center_recovery/section412_editorial_evidence/README_FA.md)؛ ۴۴ فیلد دیگر قرارداد به صورت مستقل پذیرفته نشده‌اند |
| R_KL | دقیق support-aware با R_KL^(legacy,clip) قدیمی متفاوت است | [فصل۳](../tex/chapter3.tex)، [گزارش §۴.۱۲](../code/experiments/standard_policy_comparison/r03_center_recovery/section412_editorial_evidence/README_FA.md)؛ برچسب تاریخی عدد را تغییر نده |
| R، I و BA | جدایی مراکز به‌تنهایی فشردگی نمونه و BA را تعیین نمی‌کند | [فصل۳](../tex/chapter3.tex)، [Benchmark](../code/experiments/benchmark_v1/RESULTS_FA.md) |
| Softmax و سیگموئید | ℓτ(m)=σ(−m/τ)، gτ=4τ|ℓ′τ|؛ نرمال‌سازی حساسیت گرادیان سیاست نیست | [فصل۳](../tex/chapter3.tex)، [فصل۴](../tex/chapter4.tex)، [Issue #47](https://github.com/mahhammad/My-Thesis/issues/47) |
| node/node_class | بدون تابع شروع بیرون‌گراف F، Validation قابل استنتاج خودکار نیست | [Benchmark](../code/experiments/benchmark_v1/RESULTS_FA.md)، [فصل۳](../tex/chapter3.tex) |
| ممیزی فصل۳ | ۵۹ A / ۳ B / صفر C یا D در ۶۲ ادعای متن؛ دو R توسعه اختیاری | [گزارش رسمی ۶۴ ردیفی](2026-11-14-ch3-full-scientific-editorial-audit.md)، [ممیزی مدیر](https://github.com/mahhammad/My-Thesis/issues/70#issuecomment-6869287126)؛ **MANAGER_SELF_AUDIT** ≠ شخص ثالث |
| مدیریت شاخه‌ها | شاخهٔ دائمی فقط main؛ یک شاخه و یک PR برای کار؛ امضای گیت و مالک مرج | [WORKFLOW_FA.md](WORKFLOW_FA.md)، [Issue #55 تاریخی](https://github.com/mahhammad/My-Thesis/issues/55) |

## ترتیب زمان و منع احیای استنتاج منسوخ
۱. گزارش‌های D10/D11 قدیمی ابتدا SCIENTIFIC_HOLD بودند. پس از بازسازی کران دقیق و گواه خاص §۴.۱۱، **نسخه جدید** در محدوده گیت خود معتبر است؛ حذف نسخه اولیه از تاریخچه یا تبدیل پذیرش کد V2 به گواه همه داده‌ها مجاز نیست.
۲. تاریخچه §۴.۱۰ از B01، snapshot B02، verifier دارای خطای G15/G20 و اصلاح V2 گذشت. رأی [ACCEPT_PURE_VERIFIER_FOR_INTEGRATION_REVIEW](https://github.com/mahhammad/My-Thesis/issues/49#issuecomment-6851300292) با محدودیت، **به‌تنهایی مجوز اجرای حلگر یا اثبات هر خروجی جدید نیست**؛ انتشار §۴.۱۰ نهایتاً با PR #61 و #62 انجام شد. نسخه‌های raw اولیه نباید پاک یا جایگزین شوند.
۳. انتشار §۴.۱۲ با PR #66 و انتقال نهایی به main با PR #72 و نگارش فصل سوم با PR #76 انجام شد. Draft قدیمی PR #50/#51 پس از انتشارهای جدید مرجع فعال نیست.
۴. هر ادعای جدید دربارهٔ تعمیم تجربی، تغییر زیان یا مرکز هدف به پیش‌ثبت، seed مجزا و QA جدید نیاز دارد؛ گزارش آینده را به‌جای قضیه وارد فصل ۳ نکن.
