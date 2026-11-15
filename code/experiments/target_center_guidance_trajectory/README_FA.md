# گزارش فنی پژوهشی — E-CH4-06 | مسیر هدایت مراکز در ۴۹ دور

**شناسهٔ مطالعه:** `target_center_guidance_trajectory` | **حوزه:** پایان‌نامه IMC / فصل چهارم | **منشأ محتوا:** اسناد پیش از تجمیع در commit `17e91ec99d1b`

## چکیده و دامنهٔ استناد

تحلیل مشاهده‌ای مسیر ۴۹دوری مراکز هدف و تحقق‌یافته با تأکید بر تفاوت خطای هدف، هندسهٔ مرکز، تراکم و BA؛ دورها تکرار مستقل نیستند.

این سند هم **گزارش علمی** و هم **راهنمای فنی پیاده‌سازی و شواهد** است. خروجی عددی در فایل‌های ماشین‌خوان باقی می‌ماند؛ صرفِ درج یک نتیجه در این گزارش به معنای ممیزی آن فراتر از محدودهٔ ثبت‌شده نیست.

## گزارش علمی: سؤال، پروتکل، معیارها، نتایج و محدودیت‌ها

## آزمایش ۴.۶ — هدایت گشت به مراکز هدف و تحلیل مسیر معیارهای بازنمایی

**تکمیل بازپخش معیارها (۱۰ نوامبر ۲۰۲۶):** هر ۴۹ هدف ثابت تاریخچه با همان ۱۰۰ گشت، بذر و سیاست هدایت‌شده بازپخش شد و تمام مراکز با تاریخچه تطبیق عددی داشتند. اکنون در results/metrics_full.json#replayed_fixed_targets هر ۴۹ دور دارای سه تراکم R/R_JS/R_sqrt_JS، پنج جدایی T/BT/TNL/BTNL/I، شاخص‌های حاشیه/دقت/NLL و هندسه کامل هستند. در ۴۴ دور پس از ادامه تاریخچه قدیمی، R_JS ذخیره‌شده میانگین نامتوازن نمونه‌ها بوده است؛ بازپخش مقدار **متوازن کلاس‌ها** را طبق فصل سوم مستقل محاسبه کرد. از ۴۹ دقت Validation، ۲۵ مقدار به‌صورت تاریخی ثبت شده بودند و ۲۴ مقدار در این بازپخش تازه محاسبه شدند؛ انتخاب قبلی دور چهار تغییر نکرده است. این یک بازپخشِ هدف ثابت است، نه چندبذری مستقل یا مقایسه هم‌پروتکل با Benchmark.

[تست موفق بازپخش ۴۹ دور](https://github.com/mahhammad/My-Thesis/actions/runs/42782841819) · [شکل متوازن معیارها](figures/trajectory_geometry_balanced.svg) · [نقاط دقت تاریخی و بازپخش](figures/validation_ba_replayed_49.svg).


**شناسه:** E-CH4-06 / D04 · **نوع:** تحلیل مشاهده‌ای هدایت محلی مراکز در ۴۹ دور وابسته. **وضعیت:** نتایج تاریخی بدون تغییر حفظ و بازپخش تمام ۴۹ هدف ثابت انجام شده است؛ این بازپخش شامل گشت جدید با همان سیاست و بذر، بدون یادگیری مجدد مسیر هدف یا استفاده از Test است.

**انطباق با قرارداد مشترک:** [گزارش ماشین‌خوان](results/metrics_full.json) برای همه ۴۹ دور، معیارهای مرکزمحور $T,BT,TNL,BTNL,I$، فاصله زوجی و $R_{\mathrm{JS}}$ متوازن را دارد. [بازپخش هدف‌های ثابت](results/metrics_full.json) سه سنجه تراکم $R_{\mathrm{KL}},R_{\mathrm{JS}},R_{\sqrt{\mathrm{JS}}}$ و سایر سنجه‌های تصمیم/حاشیه و دقت را نیز برای ۴۹ دور محاسبه کرده است. فقط ۲۵ مقدار BA از تاریخچه اولیه هستند؛ ۲۴ مقدار جدید بازپخش مستقلاً علامت‌گذاری شده‌اند. [فایل پوشش معیارها](results/coverage.json) این موارد را به‌صراحت «اندازه‌گیری‌نشده» ثبت می‌کند. مقادیر تازه از روی مراکز حدس زده نشده‌اند؛ با بازپخش گشت از هدف ثابت هر دور محاسبه و با مراکز تاریخی تأیید شده‌اند. این‌ها بذر مستقل یا بخش جدید Benchmark محسوب نمی‌شوند. تعریف ریاضی معیارها در [فصل سوم و قرارداد مشترک](../COMMON_METRICS_FA.md) است، نه در این گزارش.

### ۱. سؤال و فرضیه

پرسش: آیا نزدیک‌شدن مراکز تحقق‌یافته به مراکز هدف، یا افزایش جدایی مراکز، لزوماً تراکم نمونه‌ها، حاشیه و عملکرد طبقه‌بندی را بهتر می‌کند؟ فرضیه آن است که معیارهای جدایی و تراکم و فاصله از هدف ممکن است مسیر یکسانی نداشته باشند؛ این آزمایش برای مشاهده مسیر و استخراج محدودیت‌هاست، نه اثبات همگرایی.

### ۲. سیاست و محاسبه

در دور $t$، مرکز هدف کلاس $c$ را با $U_c^{(t)}$ و مرکز تحقق‌یافته پس از گشت را با $\widehat V_c^{(t)}$ نمایش می‌دهیم. برای هر گره و هر فرض کلاس، سیاست انتخاب ویژگی با تقریب حل
\[
\pi^*_{j\mid s,c}=\arg\min_{\pi\in\Delta(B_j^{(s)})}
\operatorname{JSD}\big((Q_j^{(s)})^\top\pi,U_c^{(t)}\big)
\]
با حداکثر ۴۵ گام Frank–Wolfe تعیین می‌شود. این بهینه‌سازی **محلی و یک‌گامی** است؛ مراکز واقعی از گشت **چندگامی** با همان سیاست محاسبه می‌شوند. پس از ارزیابی همان دور:
\[
U_c^{(t+1)}=(1-\eta)U_c^{(t)}+\eta\widehat V_c^{(t)},\quad \eta=0.5.
\]
آغاز هدف نزدیک یک‌داغ با $\epsilon=.08$ است. هدف $U$ را نباید با مرکز تحقق‌یافته $\widehat V$ یکی گرفت.

### ۳. داده، بذر، پروتکل و عدم نشت

| پارامتر | مقدار یا قاعده |
|---|---|
| مولد و اندازه | سه کلاس، پنج ویژگی؛ Train=۳۰۰، Validation=۹۰، یک تقسیم ثابت |
| گشت | $\beta=.4$، $\lambda=.8$، $H=12$ و **۱۰۰ گشت مونت‌کارلو برای هر نمونه** |
| بذر/سیاست | بذر گشت `20261104`، سیاست محلی کلاس‌فرضی با حداکثر ۴۵ گام Frank–Wolfe |
| هدف | آغاز نزدیک یک‌داغ؛ به‌روزرسانی هدف با $\eta=.5$ |
| تاریخچه | ۴۹ دور ثبت‌شده از ۰ تا ۴۸؛ ۲۵ BA در فایل اولیه، ۲۴ BA تازه از بازپخش هدف ثابت |
| انتخاب وضعیت شاخص | دور ۴، بهترین BA بین **همان ۲۵ دور اندازه‌گیری‌شده**، با انتخاب بر اساس Validation |
| Test تاریخی | در انتخاب دور یا ارزیابی فعلی استفاده نشده |

**تفاوت دو روش استنتاج:** در Uniform تاریخی برای هر پرس‌وجو یک بازنمایی مستقل از فرض کلاس ساخته شده؛ در Guided برای **هر فرض کلاس یک بازنمایی مجزا** ساخته و با مرکز تحقق‌یافته همان کلاس سنجیده می‌شود. برچسب واقعی Validation در تولید این مسیرها دخالت نمی‌کند. مقایسه مستقیم BA این دو روش، ارزیابی هم‌پروتکلِ اثر *صرفاً* سیاست نیست. همچنین دقت Train در Guided از کلاس واقعی مبدأ برای ساخت سیاست استفاده می‌کند و شاخص استنتاج بدون برچسب نیست.

### ۴. معیارهای گزارش

- $I$: معیار اطلاعاتی جدایی مراکز **تحقق‌یافته** آموزش.
- $R_{\mathrm{JS}}$: پراکندگی درون‌کلاسی متوازن آموزش حول مراکز واقعی، نه فاصله تا مرکز هدف. در این مطالعه بازنمایی آموزش Guided به کلاس مبدأ شرطی است.
- $G=\frac1K\sum_c\operatorname{JSD}(U_c,\widehat V_c)$: خطای هدف–تحقق؛ کوچک‌شدن آن تضمین جدایی مراکز نیست.
- BA اعتبارسنجی: میانگین Recall کلاس‌ها؛ ۲۵ مقدار از تاریخچه اولیه و ۲۴ مقدار به‌صورت مستقل از بازپخش هدف‌های ثابت محاسبه شده‌اند. هیچ‌یک صفر یا درون‌یابی نیست.
- NLL، KL، جذر JSD و حاشیه ابتدا فقط برای سه وضعیت دارای بازنمایی ذخیره‌شده قابل بازتحلیل بودند؛ اکنون بازپخش هدف‌های ثابت این معیارها را برای هر ۴۹ دور تکمیل کرده است.

دورهای یک مسیر وابسته‌اند؛ میانگین و SD بین‌بذری در این گزارش وجود ندارد.

### ۵. جدول‌ها و نمودارهای معتبر

[جدول سه وضعیت منتخب](tables/three_cases.tex)، [جدول دورهای منتخب با تراکم متوازن](tables/selected_rounds_balanced.tex)، [نمودار I و R متوازن ۴۹دوری](figures/trajectory_geometry_balanced.tex) و [نمودار ۲۵ BA تاریخی در کنار ۲۴ مقدار جدید](figures/validation_ba_replayed_49.tex) منابع چاپی هستند. فایل‌های SVG متناظر و نسخه‌های تاریخیِ نامتوازن بدون حذف برای ممیزی نگهداری می‌شوند.

| وضعیت | $I$ | $R_{\mathrm{JS}}$ | خطای هدف/واقعی $G$ | BA اعتبارسنجی |
|---|---:|---:|---:|---:|
| Guided، دور ۰ | ۰٫۲۶۸۷۳ | ۰٫۰۱۳۲۶ | ۰٫۰۷۹۰۴۹ | ۷۲٫۶۱٪ |
| Guided، دور ۴ | ۰٫۱۴۱۱۱ | ۰٫۰۰۳۷۶ | ۰٫۰۰۲۴۲۸ | ۸۳٫۸۳٪ |
| Guided، دور ۲۰ | ۰٫۰۰۸۶۰ | ۰٫۰۰۱۶۸ | ۰٫۰۰۰۱۲۳ | ۴۷٫۵۳٪ |
| Guided، دور ۴۸ | ۰٫۰۰۰۷۹ | ۰٫۰۰۱۵۹ | ۰٫۰۰۰۰۰۱ | ۳۴٫۰۴٪ |

**مقایسه دو دور Guided با پروتکل مشابه:** از دور ۰ تا ۴، $I$ کاهش یافته و هم‌زمان پراکندگی $R_{\mathrm{JS}}$ هم کم شده است؛ BA ثبت‌شده ۱۱٫۲۲ واحد درصد بالا رفته است. بهبود BA را نمی‌توان صرفاً نتیجه کاهش $R$ دانست، زیرا مراکز، سیاست و اثر نمونه‌گیری مونت‌کارلو هم تغییر کرده‌اند. این یافته در امتداد آزمایش ۴.۵ نشان می‌دهد که بهبود شاخص جدایی، به‌تنهایی معیار کافی برای بهترشدن عملکرد نیست.

**وضعیت Uniform تاریخی:** BA=۸۴٫۸۶٪، $I=0.01818$ و $R_{\mathrm{JS}}=0.00388$؛ اما این حالت با Guided از لحاظ قاعده استنتاج متفاوت است و نباید به‌عنوان برنده مقایسه مستقیم معرفی شود.

در دور ۴۸، فاصله هدف–واقعی بسیار کوچک می‌شود، اما اطلاعات مراکز تقریباً به صفر رسیده و BA ثبت‌شده ۳۴٫۰۴٪، نزدیک مرجع ۳۳٫۳۳٪ برای حدس یکنواخت سه‌کلاسه است. **نزدیک‌شدن به هدف تضمین نمی‌کند خودِ هدف برای طبقه‌بندی مناسب مانده باشد.** این الگو اثبات همگرایی یا فروپاشی اجتناب‌ناپذیر در همه تنظیم‌ها نیست.

از سوی دیگر در مقایسه دو وضعیت Guided، NLL ساده KL اعتبارسنجی از ۱٫۰۰۲۸۹ در دور صفر به ۱٫۰۴۹۱۰ در دور چهار افزایش می‌یابد، درحالی‌که BA بالا رفته است. برای جزئیات [گزارش مقادیر کامل](results/detailed_three_cases.json) مراجعه شود.


#### خروجی کامل معیارها و پوشش تاریخی

[گزارش ماشین‌خوان کامل](results/metrics_full.json) **تمام ۴۹ مجموعه مراکز تحقق‌یافته** را با معیارهای $T,BT,TNL,BTNL,I$، ماتریس‌های کامل JSD و جذر JSD میان مراکز، کمینه، میانگین و نزدیک‌ترین فاصله برای هر کلاس ثبت می‌کند. برای ۴۹ دور، $R_{\mathrm{JS}}$ تاریخی آموزش نیز موجود است؛ در ۴۴ دورِ بعد از ادامه تاریخچه، میانگین آن نامتوازن بوده و در بازپخش با قرارداد متوازن اصلاح شده است. دقت اعتبارسنجیِ تاریخچه اولیه در ۲۵ دور ذخیره شده و ۲۴ دور دیگر فقط در خروجی بازپخش گزارش شده است.

برای سه وضعیت دارای بازنمایی‌های نمونه‌ای ذخیره‌شده (Uniform، Guided دور ۰ و Guided دور ۴)، گزارشگر مشترک تمام سه تراکم $R=R_{\mathrm{KL}}$، $R_{\mathrm{JS}}$، $R_{\sqrt{\mathrm{JS}}}$ را هم روی Train و هم Validation و همراه معیارهای تصمیم، حاشیه و NLL باز‌محاسبه و با گزارش اولیه تطبیق داده است؛ [CSV مشترک](results/metrics_summary.csv) و [جدول سه معیار تراکم](tables/complete_metrics.tex) نیز فراهم‌اند.

برای ۴۷ دور Guided دیگر، داده خام بازنمایی تک‌نمونه‌ای در تاریخچه اصلی ثبت نشده بود و بازیابی معیارها از خود مراکز ممکن نیست. این کمبود اکنون با **بازپخش هدف‌های ثابت** و تطبیق مراکز تاریخی، نه با برون‌یابی یا مقدار صفر، جبران شده است. گزارش مستقل در `results/metrics_full.json#replayed_fixed_targets` همه سنجه‌های عمومی Train/Validation برای ۴۹ دور را دارد. در دورهای ۵ تا ۴۸، مقدار قدیمی $R_{\mathrm{JS}}$ به‌صورت میانگین نامتوازن نمونه‌ها ثبت شده و با مقدار متوازن قرارداد فصل سوم تفاوت دارد؛ هر دو برای ممیزی و در شکل جدید فقط مقدار متوازن به کار رفته است. مقادیر دقت Train در Guided از سیاست شرطی بر *برچسب معلوم مبدأ* حاصل می‌شوند و معیار عملکرد قابل‌تعمیمِ یک طبقه‌بند بدون برچسب نیستند.


### ۶. مسیر فایل‌ها و بازتولید

تمام **۱۱ فایل خام نتایج** از [آرشیو اصلی](https://github.com/mahhammad/My-Thesis/blob/a8c9027460e538fa04b8aab2f762eb4ba4c31e3d/code/experiments/archive/center_matching_comparison/README_FA.md) با **Git blob SHA یکسان** به `results/` کپی شدند. تاریخچه ۴۹ دور، سه وضعیت، معیارهای هندسه، CSVها و بازنمایی‌های هر نمونه در پوشه موجودند؛ اصل آرشیو حذف نشده است.

- [تاریخچه ۴۹ دور](results/history_00_48.json)، [هندسه ۴۹ دور](results/center_geometry_49.json)، [شاخص‌ها به تفکیک دور](results/trajectory_metrics.csv).
- [سه وضعیت و مراکز](results/three_cases.json)، [معیارهای دقیق نمونه‌به‌نمونه](results/detailed_three_cases.json) و سه فایل بازنمایی در `results/`.
- [کد بازتحلیل](analyze.py)، [کد معیارهای تفصیلی](detailed_metrics.py)، [تست‌های بازپخش آمار](test_analyze.py) و [تست‌های معیار](test_detailed_metrics.py) نیز از آرشیو حفظ شدند.
- [مانیفست شواهد](results/provenance.json) و [آزمون گزارش](test_report_assets.py) جداسازی مقدار اندازه‌گیری‌نشده از صفر و صحت عددی جدول/نمودار را بررسی می‌کنند.

آزمون سبک (بدون محاسبه دوباره گشت):

`python -m unittest discover -s code/experiments/target_center_guidance_trajectory -p 'test_*.py' -v`

بازپخش هدف ثابت برای ۴۹ دور از فایل `replay_fixed_targets.py` انجام شده و در GitHub Actions با کنترل تطبیق مراکز، ۲۵ دقت تاریخی و تفاوت تجمیع $R_{\mathrm{JS}}$ موفق بوده است. این کار بازآموزی، تغییر هدف‌های تاریخی یا بازتولید مسیر هدف از ابتدا نیست و Test را لمس نمی‌کند.

### ۷. نتیجه و محدودیت

این آزمایش شاهدی مشاهده‌ای برای تفکیک «نزدیکی به هدف»، «جدایی»، «پراکندگی» و «دقت» ارائه می‌کند. این چهار مفهوم یکسان نیستند. پروتکل گشت و استنتاج این مطالعه با آزمایش ۴.۵ (امید دقیق/سیاست مشترک) یکسان نیست؛ از مقایسه بی‌قید BA آنها نتیجه علمی استخراج نمی‌کنیم. نتایج مربوط به یک مولد و یک بذرند و برای ادعای تعمیم یا قضاوت معناداری اختلاف‌ها به تکرار مستقل نیاز دارند.


## راهنمای اجرایی و توضیحات تکمیلیِ نسخهٔ قبلی

## E-CH4-06 | هدایت گشت به مراکز هدف و تحلیل مسیر معیارهای بازنمایی

**تکمیل بازپخش معیارها (۱۰ نوامبر ۲۰۲۶):** هر ۴۹ هدف ثابت تاریخچه با همان ۱۰۰ گشت، بذر و سیاست هدایت‌شده بازپخش شد و تمام مراکز با تاریخچه تطبیق عددی داشتند. اکنون در results/metrics_full.json#replayed_fixed_targets هر ۴۹ دور دارای سه تراکم R/R_JS/R_sqrt_JS، پنج جدایی T/BT/TNL/BTNL/I، شاخص‌های حاشیه/دقت/NLL و هندسه کامل هستند. در ۴۴ دور پس از ادامه تاریخچه قدیمی، R_JS ذخیره‌شده میانگین نامتوازن نمونه‌ها بوده است؛ بازپخش مقدار **متوازن کلاس‌ها** را طبق فصل سوم مستقل محاسبه کرد. از ۴۹ دقت Validation، ۲۵ مقدار به‌صورت تاریخی ثبت شده بودند و ۲۴ مقدار در این بازپخش تازه محاسبه شدند؛ انتخاب قبلی دور چهار تغییر نکرده است. این یک بازپخشِ هدف ثابت است، نه چندبذری مستقل یا مقایسه هم‌پروتکل با Benchmark.

[تست موفق بازپخش ۴۹ دور](https://github.com/mahhammad/My-Thesis/actions/runs/42782841819) · [شکل متوازن معیارها](figures/trajectory_geometry_balanced.svg) · [نقاط دقت تاریخی و بازپخش](figures/validation_ba_replayed_49.svg).


**قرارداد عمومی سنجش:** [فصل سوم](../COMMON_METRICS_FA.md)؛ [گزارش کامل](results/metrics_full.json) · [پوشش و تمایز منشأ](results/coverage.json). همه ۴۹ دور در بازپخشِ هدف‌های ثابت، سه تراکم، پنج جدایی و دقت Train/Validation دارند؛ ۲۵ دقت اعتبارسنجی از تاریخچه اصلی و ۲۴ مورد صرفاً حاصل بازپخش‌اند. R_JS متوازن از نو محاسبه شده و مقدار تاریخیِ نامتوازن ۴۴ دور نیز برای ممیزی حفظ شده است.

**پوشه رسمی آزمایش ۴.۶.** این پوشه داده‌های مطالعه تاریخی را بدون تغییر نگه می‌دارد و یک بازپخش مجزای هدف‌های ثابت را نیز همراه کنترل تطبیق مراکز و تمایز مقدار تاریخی/جدید گزارش می‌کند. [گزارش علمی فارسی](README_FA.md) پروتکل، محدودیت‌ها و نتایج را با همان استاندارد بخش‌های ۴.۳ تا ۴.۵ توضیح می‌دهد.

### نمای کلی شواهد

۴۹ دور از صفر تا ۴۸ ثبت شده‌اند؛ تمام ۴۹ دور معیارهای هندسی مرکز دارند. **BA اعتبارسنجی در تاریخچه اصلی فقط در ۲۵ دور اندازه‌گیری شده بود** و ۲۴ مقدار دیگر از بازپخش آمده است. هیچ مقداری صفر/درون‌یابی نشده است. دور چهار بهترین BA در همین ۲۵ دور است و با Validation انتخاب شده است. سه وضعیت Uniform، Guided دور صفر و Guided دور چهار به‌طور تفصیلی تحلیل شده‌اند.

مقایسه Uniform و Guided از نظر پروتکل استنتاج یکسان نیست: Uniform یک بازنمایی مشترک و Guided یک بازنمایی جدا برای هر فرض کلاس دارد. در نتیجه این مطالعه یک **آزمایش مشاهده‌ای برای فهم هندسه و معیارهاست**، نه انتخاب برنده در یک رقابت هم‌پروتکل.

### مسیر فایل‌ها

| مسیر | نقش |
|---|---|
| [REPORT_FA.md](README_FA.md) | **گزارش رسمی و تفسیر علمی** |
| [results/provenance.json](results/provenance.json) | شناسه و SHA تمام ۱۱ فایل خامِ کپی‌شدهٔ بدون تغییر از آرشیو |
| [results/history_00_48.json](results/history_00_48.json) | مراکز هدف و تحقق‌یافته، اطلاعات گشت و BAهای واقعاً ثبت‌شده |
| [results/trajectory_metrics.csv](results/trajectory_metrics.csv) | مسیر کامل $I$، $R_{\mathrm{JS}}$، فاصله هدف–واقعی و BAهای موجود |
| [results/center_geometry_49.json](results/center_geometry_49.json) | ماتریس‌های مراکز، معیارهای جدایی و هندسه همه ۴۹ دور |
| [results/three_cases.json](results/three_cases.json) | Uniform، Guided دور صفر و دور چهار با پروتکل و مراکزشان |
| [results/detailed_three_cases.json](results/detailed_three_cases.json) | NLL/تراکم/حاشیه و سایر معیارهای دقیق سه وضعیت |
| [results/representations_uniform.json](results/representations_uniform.json) | بازنمایی نمونه‌های Uniform |
| [results/representations_guided_round0.json](results/representations_guided_round0.json)، [results/representations_guided_round4.json](results/representations_guided_round4.json) | بازنمایی تک‌نمونه‌ای دو وضعیت Guided |
| [tables/three_cases.tex](tables/three_cases.tex) | جدول اصلی سه وضعیت و caveat درباره پروتکل |
| [tables/selected_rounds_balanced.tex](tables/selected_rounds_balanced.tex) | جدول چاپی با $R_{\mathrm{JS}}$ متوازن و BA تاریخی؛ دورهای فاقد اندازه‌گیری در تاریخچه اولیه مشخص شده‌اند |
| [figures/trajectory_geometry_balanced.tex](figures/trajectory_geometry_balanced.tex) / [SVG](figures/trajectory_geometry_balanced.svg) | روند $I$ و $R_{\mathrm{JS}}$ متوازن در ۴۹ دور بازپخش |
| [figures/validation_ba_replayed_49.tex](figures/validation_ba_replayed_49.tex) / [SVG](figures/validation_ba_replayed_49.svg) | ۲۵ مقدار BA تاریخی با نشان پر، ۲۴ مقدار بازپخش جدید با نشان توخالی |
| [analyze.py](analyze.py)، [detailed_metrics.py](detailed_metrics.py) | اسکریپت‌های تحلیل از آرشیو، بدون تکرار گشت |
| [test_analyze.py](test_analyze.py)، [test_detailed_metrics.py](test_detailed_metrics.py)، [test_report_assets.py](test_report_assets.py) | تست‌های عددی و کنترل شواهد |

تمام ۱۱ فایل خام با **همان Git blob SHA** در [آرشیو اصلی](https://github.com/mahhammad/My-Thesis/blob/a8c9027460e538fa04b8aab2f762eb4ba4c31e3d/code/experiments/archive/center_matching_comparison/README_FA.md) نیز باقی مانده‌اند. ابزار بازاجرای مونت‌کارلوی تاریخی [reproduce.py](https://github.com/mahhammad/My-Thesis/blob/a8c9027460e538fa04b8aab2f762eb4ba4c31e3d/code/experiments/archive/center_matching_comparison/reproduce.py) در آرشیو نگه داشته شده و در این مرحله استفاده نشده است.

### آزمایش بدون محاسبه سنگین

از ریشه مخزن:

`python -m unittest discover -v -s code/experiments/target_center_guidance_trajectory -p 'test_*.py'`

تست‌های سبک تداوم مراکز هدف، فرمول‌های JSD، گزینش دور چهار از ۲۵ مقدار تاریخی و انطباق جدول‌ها و شکل‌های تاریخی را کنترل می‌کنند. آزمون تکمیلی `test_replay_presentation.py` نیز هر ۴۹ دورِ بازپخش‌شده، تجمیع متوازن $R_{\mathrm{JS}}$، ۴۴ اختلاف قدیمی و ۲۵/۲۴ نقطه شکل جدید را کنترل می‌کند؛ این آزمون‌ها گشت یا آموزش جدید اجرا نمی‌کنند.

[الگوی مشترک گزارش‌های فصل چهارم](../EXPERIMENT_REPORT_STANDARD_FA.md) برای آزمایش‌های بعدی نیز مبناست.


## نقشهٔ دقیق فایل‌ها و نقش هرکدام

این جدول، پرونده‌های فعلی را در زمان تجمیع معرفی می‌کند. برای پرونده‌های داده‌ای، مقادیر و شناسهٔ اجرا در خود JSON/CSV/NPZ ثبت شده‌اند؛ فایل Markdown مستقلی برای هر خروجی لازم نیست.

| فایل یا مسیر | کاربرد فنی / پژوهشی |

|---|---|

| [`analyze.py`](analyze.py) | اسکریپت بازپخش/تحلیل/راستی‌آزمایی خروجی‌های ذخیره‌شده |

| [`backfill_common_metrics.py`](backfill_common_metrics.py) | ابزار محاسبه، تبدیل یا استخراج داده |

| [`detailed_metrics.py`](detailed_metrics.py) | ابزار محاسبه، تبدیل یا استخراج داده |

| [`figures/observed_validation_ba.svg`](figures/observed_validation_ba.svg) | نمودار برداری برای مشاهده و ممیزی بصری |

| [`figures/observed_validation_ba.tex`](figures/observed_validation_ba.tex) | جدول، نمودار یا قطعهٔ TeX مورد استفاده در پایان‌نامه |

| [`figures/trajectory_geometry.svg`](figures/trajectory_geometry.svg) | نمودار برداری برای مشاهده و ممیزی بصری |

| [`figures/trajectory_geometry.tex`](figures/trajectory_geometry.tex) | جدول، نمودار یا قطعهٔ TeX مورد استفاده در پایان‌نامه |

| [`figures/trajectory_geometry_balanced.svg`](figures/trajectory_geometry_balanced.svg) | نمودار برداری برای مشاهده و ممیزی بصری |

| [`figures/trajectory_geometry_balanced.tex`](figures/trajectory_geometry_balanced.tex) | جدول، نمودار یا قطعهٔ TeX مورد استفاده در پایان‌نامه |

| [`figures/validation_ba_49_unified.tex`](figures/validation_ba_49_unified.tex) | جدول، نمودار یا قطعهٔ TeX مورد استفاده در پایان‌نامه |

| [`figures/validation_ba_replayed_49.svg`](figures/validation_ba_replayed_49.svg) | نمودار برداری برای مشاهده و ممیزی بصری |

| [`figures/validation_ba_replayed_49.tex`](figures/validation_ba_replayed_49.tex) | جدول، نمودار یا قطعهٔ TeX مورد استفاده در پایان‌نامه |

| [`replay_fixed_targets.py`](replay_fixed_targets.py) | اسکریپت بازپخش/تحلیل/راستی‌آزمایی خروجی‌های ذخیره‌شده |

| [`results/center_geometry_49.csv`](results/center_geometry_49.csv) | دادهٔ جدولی اجرای آزمایش یا ورودی جدول‌های پایان‌نامه |

| [`results/center_geometry_49.json`](results/center_geometry_49.json) | داده، مقادیر ذخیره‌شده یا شواهد ماشین‌خوان |

| [`results/coverage.json`](results/coverage.json) | مانیفست پوشش معیارها، مقادیر موجود و دلایل عدم کاربرد |

| [`results/detailed_three_cases.csv`](results/detailed_three_cases.csv) | دادهٔ جدولی اجرای آزمایش یا ورودی جدول‌های پایان‌نامه |

| [`results/detailed_three_cases.json`](results/detailed_three_cases.json) | داده، مقادیر ذخیره‌شده یا شواهد ماشین‌خوان |

| [`results/history_00_48.json`](results/history_00_48.json) | داده، مقادیر ذخیره‌شده یا شواهد ماشین‌خوان |

| [`results/metrics_full.json`](results/metrics_full.json) | خروجی عددی ساخت‌یافته و معیارهای اندازه‌گیری‌شده |

| [`results/metrics_summary.csv`](results/metrics_summary.csv) | خلاصه عددی/جدول معیارها و تجمیع نتایج |

| [`results/provenance.json`](results/provenance.json) | هویت منابع، تنظیمات و کنترل منشأ داده |

| [`results/representations_guided_round0.json`](results/representations_guided_round0.json) | داده، مقادیر ذخیره‌شده یا شواهد ماشین‌خوان |

| [`results/representations_guided_round4.json`](results/representations_guided_round4.json) | داده، مقادیر ذخیره‌شده یا شواهد ماشین‌خوان |

| [`results/representations_uniform.json`](results/representations_uniform.json) | داده، مقادیر ذخیره‌شده یا شواهد ماشین‌خوان |

| [`results/three_cases.csv`](results/three_cases.csv) | دادهٔ جدولی اجرای آزمایش یا ورودی جدول‌های پایان‌نامه |

| [`results/three_cases.json`](results/three_cases.json) | داده، مقادیر ذخیره‌شده یا شواهد ماشین‌خوان |

| [`results/trajectory_metrics.csv`](results/trajectory_metrics.csv) | خلاصه عددی/جدول معیارها و تجمیع نتایج |

| [`tables/complete_metrics.tex`](tables/complete_metrics.tex) | جدول، نمودار یا قطعهٔ TeX مورد استفاده در پایان‌نامه |

| [`tables/selected_center_geometry.tex`](tables/selected_center_geometry.tex) | جدول، نمودار یا قطعهٔ TeX مورد استفاده در پایان‌نامه |

| [`tables/selected_density.tex`](tables/selected_density.tex) | جدول، نمودار یا قطعهٔ TeX مورد استفاده در پایان‌نامه |

| [`tables/selected_rounds.tex`](tables/selected_rounds.tex) | جدول، نمودار یا قطعهٔ TeX مورد استفاده در پایان‌نامه |

| [`tables/selected_rounds_balanced.tex`](tables/selected_rounds_balanced.tex) | جدول، نمودار یا قطعهٔ TeX مورد استفاده در پایان‌نامه |

| [`tables/selected_rounds_unified.tex`](tables/selected_rounds_unified.tex) | جدول، نمودار یا قطعهٔ TeX مورد استفاده در پایان‌نامه |

| [`tables/selected_validation.tex`](tables/selected_validation.tex) | جدول، نمودار یا قطعهٔ TeX مورد استفاده در پایان‌نامه |

| [`tables/three_cases.tex`](tables/three_cases.tex) | جدول، نمودار یا قطعهٔ TeX مورد استفاده در پایان‌نامه |

| [`tables/validation_ba_49_unified.tex`](tables/validation_ba_49_unified.tex) | جدول، نمودار یا قطعهٔ TeX مورد استفاده در پایان‌نامه |

| [`test_analyze.py`](test_analyze.py) | آزمون خودکار صحت کد، داده و قرارداد گزارش |

| [`test_detailed_metrics.py`](test_detailed_metrics.py) | آزمون خودکار صحت کد، داده و قرارداد گزارش |

| [`test_replay_presentation.py`](test_replay_presentation.py) | آزمون خودکار صحت کد، داده و قرارداد گزارش |

| [`test_report_assets.py`](test_report_assets.py) | آزمون خودکار صحت کد، داده و قرارداد گزارش |

| [`test_unified_narrative.py`](test_unified_narrative.py) | آزمون خودکار صحت کد، داده و قرارداد گزارش |

| `README_FA.md` | همین گزارش فنی تجمیعی، پروتکل و نقشهٔ شواهد |

## سیاست حفظ تاریخچه و پیوندهای قدیمی

فایل‌های Markdown پیشین (`REPORT_FA.md`) پس از تجمیع از checkout فعال حذف می‌شوند؛ متن آن‌ها با شرح منشأ در همین سند آمده و Git blob اصلی در commit [`17e91ec99d1b`](https://github.com/mahhammad/My-Thesis/tree/17e91ec99d1bf082ef87e658d0643137b1e7b6fe/code/experiments/target_center_guidance_trajectory/) قابل بازیابی است. ارجاعات تاریخی کامنت‌های Issue اصلاح‌پذیر نیستند و باید با نسخهٔ قفل‌شدهٔ Git تفسیر شوند.
