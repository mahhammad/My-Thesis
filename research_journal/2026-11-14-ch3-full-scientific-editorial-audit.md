# ممیزی جامع علمی، ریاضی، نگارشی و انتقال دانش فصل سوم IMC
**مأموریت:** `CH3-FULL-SCIENTIFIC-EDITORIAL-AUDIT-01` — نقش `LTX-EDITOR-CH3-01`  
**تاریخ:** ۲۰۲۶-۱۱-۱۴  
**مبدأ:** `main@7829e15bc6c754b73a420d6ccdbaf0d3de380180`  
**شاخه واحد:** `editorial/ch3-language-reference-audit-01`؛ **PR واحد:** [#76](https://github.com/mahhammad/My-Thesis/pull/76) به `main` (Draft)  
**وضعیت تحویل:** `READY_FOR_MANAGER_SCIENTIFIC_REVIEW` **مشروط به گیت‌های بازِ زیر؛ نه رأی پذیرش برای ادغام**.

## خلاصه و روش ممیزی
این ممیزی از سند و فرمول شروع کرده و ادعای «همه چیز از قبل صحیح بود» را فرض نگرفت: مسیرهای `chapter3.tex`، `appendix_chapter3.tex`، `notations.tex` و سازگاری با `chapter4.tex`، همهٔ **۳۰ ژورنال موجود در شاخه** و **دو ژورنال نظریِ پژوهشگر ۲ فقط در کامیت‌های آرشیوی** بررسی شدند؛ Issueهای #32/#47/#49/#54/#58/#59/#60/#67، گزارش‌های §۴.۱۰/۴.۱۱/۴.۱۲ و تغییرات PR قبلی در تعیین منشأ و تاریخ اعتبار استفاده شدند. آرشیوهای شعبه‌ای منسوخ معیار علمی نیستند. ارزیابی‌ها با بازاشتقاق هویت‌های کلیدی، خواندن استدلال‌های پیوست، کنترل حالت‌های مرزی و آزمون‌های بسیار کوچک مستقل پشتیبانی شدند؛ **از این کنترل‌ها نمی‌توان ادعای بازاثبات ماشینی همه گزاره‌ها یا CI موفق گرفت**.

**ماتریس ادعا: 64 مدخل — A=59، B=3، C=2، D=0.** B یعنی اصل معتبر بود و توضیح/ارجاع اصلاح شد؛ C دو **سؤال باز/تعمیمِ ادعانشده** هستند، نه قضایای غلطِ چاپ‌شده. دامنهٔ پذیرش به فرض‌های ستون مربوط محدود است. هیچ رأی C به PASS تبدیل نشده است.

## A — ماتریس جامع ادعا، فرض و اثبات

| ID | ادعا/تعریف | محل در فصل | فرض‌های حیاتی | اثبات/منشأ | کنترل مستقل و دامنه | رأی |
|---|---|---|---|---|---|---|
| `A01` | فضای ویژگی و فقدان داده | `eq:ch3-feature-space; eq:ch3-feature-samples` | A_X فقط ویژگی حاضر؛ صفر ≠ گمشده | `تعریف اصلی؛ app:ch3-policy-center-markov` | تعریف و رابطه هم‌ارزی عضویت کنترل شد | **A** |
| `A02` | اتصال‌پذیری ساختاری گراف | `eq:ch3-connectivity` | افراز غیرتهی و اشتراک ویژگی | `app:ch3-policy-center-markov` | اتصال اولیه، نه لزوماً زنجیره پس از حذف مبدأ | **A** |
| `A03` | وزن یال عمودی و مثبت‌بودن | `eq:ch3-edge-weight` | متریک هر ویژگی؛ κ مثبت | `تعریف گراف و eq:ch3-vertical-transition` | وزن مجاز و نرمال‌سازی سازگار | **A** |
| `A04` | هسته مبدأحذف و ناتهی‌بودن مقصد | `eq:ch3-vertical-transition` | هر نمونه ≥۱ ویژگی؛ برای هر b، |D_b|≥۳ | `eq:app-ch3-feasibility-Bellman` | حذف i و s و مثبت‌بودن مخرج کنترل شد | **A** |
| `A05` | سیاست افقی، ماسک و طوقه مجاز | `eq:ch3-full-transition; eq:ch3-feature-eligibility` | احتمالات مجاز مجموع یک | `app:ch3-matrix-walk-proofs` | انتقال نامجاز صفر و گام افقی/عمودی تفکیک شده | **A** |
| `B01` | وزن گشت هندسی نامحدود | `eq:ch3-h-alpha` | β در (۰،۱]؛ λ در (۰،۱]؛ توقف مستقل | `prop:app-ch3-alpha` | جابه‌جایی جمع‌های نامنفی و نرمال‌شدن ضرایب | **A** |
| `B02` | تفاوت امیدِ نسبت و نسبتِ امید | `eq:ch3-walk-vector; eq:ch3-h-alpha` | نرمال‌سازی هر گشت جداگانه | `app:ch3-weighted-walk-proof` | تعبیر صحیح و عدم جایگزینی با سری ساده | **A** |
| `B03` | ضرایب گشت سقف‌دار | `eq:ch3-capped-alpha` | L_H=min(L,H)، H≥۱ | `app:ch3-matrix-walk-proofs` | جرم باقیمانده در p_H جمع می‌شود | **A** |
| `B04` | ماتریس‌های انتقال سطری تصادفی | `eq:ch3-block-factorization` | ماسک و وجود مقصد | `app:ch3-matrix-walk-proofs` | تجزیه افقی و عمودی؛ تمایز گام نخست | **A** |
| `B05` | محاسبه امید پیش‌رونده و پس‌رونده | `eq:ch3-matrix-forward-recursion; eq:ch3-matrix-backward-recursion` | P_s و ρ_s معتبر | `app:ch3-matrix-walk-proofs` | هم‌ارزی با بازکردن بازگشت به سری محدود | **A** |
| `B06` | کران TV تقریب با r گام | `eq:ch3-matrix-truncation-bound` | r<H؛ ضرایب نامنفی | `app:ch3-matrix-walk-proofs` | جرم حذف و کران دم هندسی برقرار | **A** |
| `B07` | عمق مؤثر و محدودیت β،λ | `eq:ch3-effective-depth` | توزیع طول و مشاهده وزن‌دار | `app:ch3-matrix-walk-proofs` | طول واقعی با وزن مشاهده یکسان نیست | **A** |
| `B08` | یکتایی بازنماییِ سیاست ثابت | `eq:ch3-policy-center-unique-walk` | گراف/سیاست/H ثابت | `app:ch3-policy-center-markov` | وجود ماتریس سطری تصادفی؛ نیازمند ارگودیک نبودن | **A** |
| `B09` | نایکسانی همگرایی سری و توان ماتریس | `eq:ch3-policy-center-unique-walk` | مقادیر α نامنفی با مجموع یک | `app:ch3-policy-center-markov` | مثال نقض حالت چندکلاسه کاهش‌یافته موجود | **A** |
| `B10` | مانایی و همگرایی زنجیره بازآغاز | `eq:ch3-sequential-restart-stationary` | ۰<β<۱؛ حالت‌های قابل وصول | `app:ch3-policy-center-sequential-restart` | جرم o_s=β/(۱+β) و بازگشت روی قطعه جداست | **A** |
| `B11` | میانگین قطعات ≠ اشغال مانا | `eq:ch3-sequential-episode-average` | بازآغاز مستقل؛ وزن λ | `app:ch3-policy-center-sequential-restart` | امیدِ میانگین نرمال‌ و نسبت امیدها تمایز دارند | **A** |
| `B12` | مرتبه محاسباتی موتور عامل‌بندی‌شده | `eq:ch3-graph-size` | نمایش تنک عامل‌ها؛ گراف مرجع ثابت | `app:ch3-matrix-walk-proofs` | برآورد شرطی نظری، نه اندازه‌گیری سرعت همه پیاده‌سازی‌ها | **A** |
| `C01` | ماتریس برچسب و توزیع همسایگی | `eq:ch3-Y-script` | ماتریس Y صرفاً برچسب مرجع | `تعریف اصلی و app:ch3-matrix-walk-proofs` | Y_i خود نمونه مبدأ از خروجی حذف می‌شود | **A** |
| `C02` | مرکز کلاس به‌صورت میانگین نمونه‌ها | `eq:ch3-class-center` | کلاس غیرتهی؛ بازنمایی‌های نرمال | `app:ch3-mixture-likelihood` | میانگین توزیع معتبر؛ نه بهینه هر متریک | **A** |
| `C03` | بهینگی نماینده مشترک در KL/NLL | `eq:ch3-mixture-likelihood-optimum` | وزن نرمال گروه؛ حمایت توزیع‌ها | `eq:ch3-mixture-KL-general` | هویت میانگین KL و MLE کنترل شد | **A** |
| `C04` | KL دقیق و clipping تاریخی | `par:ch3-KL-legacy-clip` | قرارداد مؤلفه صفر و تکیه‌گاه | `app:ch3-mixture-likelihood` | ارجاع معنایی و تفکیک قرارداد تثبیت شد | **B** |
| `C05` | R و هویت آنتروپی مخلوط | `eq:ch3-R-entropy` | مرکز میانگین اعضای همان کلاس | `prop:app-ch3-entropy-mixture` | R=H(مرکز)-میانگین H؛ غیرمنفی | **A** |
| `C06` | کران R و شرط برابری | `eq:ch3-R-KL-bounds` | وزن غیرمنفی؛ اندازه کلاس | `app:ch3-compactness-entropy-relations` | R_c≤log min(K,n_c) و تساوی قابل تحلیل | **A** |
| `C07` | JSD و متریک جذر آن | `eq:ch3-JSD-definition; eq:ch3-js-distance` | توزیع روی سیمپلکس و log طبیعی | `app:ch3-js-error-proof` | ایزومتری هیلبرتی/نامساوی مثلثی کنترل شد | **A** |
| `C08` | نگاشت Φ و شباهت کسینوسی JS | `eq:ch3-js-Hilbert-map; eq:ch3-js-similarity` | Φ در L² تعریف‌شده | `eq:app-ch3-js-Hilbert-distance; eq:app-ch3-js-cosine` | نورم ثابت و کران‌های شباهت کنترل شد | **A** |
| `C09` | رأس دورتر از توزیع دلخواه | `eq:ch3-js-farthest-vertex` | JSD محدب در Q | `eq:app-ch3-js-farthest-vertex-convexity` | وجود دست‌کم یک رأس؛ نه یکتایی رأس | **A** |
| `C10` | R_JS نمونه–مرکز و R_B | `eq:ch3-R-JS-KL-midpoint-identity` | میانگین حسابی ثابت مراکز | `eq:app-ch3-R-JS-midpoint-proof` | R_JS=R_KL/2-R_B؛ ضریب ۱/۲ درست است | **A** |
| `C11` | هویت زوج‌های مرتب JS | `eq:ch3-R-JS-pairwise-entropy-identity` | همه n_c² زوج مرتب، شامل خودی | `eq:app-ch3-R-JS-pairwise-proof` | R_JSpair=R_KL-R_Bpair | **A** |
| `C12` | تفاوت میانگین جذر با جذر میانگین | `eq:ch3-R-JS-forms` | JSD≥۰ | `subsec:ch3-within-class-compactness` | استنتاج Jensen جهت صحیح دارد | **A** |
| `C13` | T/BT/TNL/BTNL و جهت بهینه‌سازی | `eq:ch3-T-on-V; eq:ch3-BT-on-V` | وزن Ω نرمال | `app:ch3-four-criteria-comparison` | بیشینه خطی؛ کمینه لگاریتمی؛ مقایسه‌ها درست | **A** |
| `C14` | کران BT و برابری جایگشتی | `eq:ch3-BT-on-V` | هر سطر V توزیع؛ Ω≥۰ جمع یک | `app:ch3-BT-bound-proof` | میانگین امتیاز جایگشت‌ها ۱/K | **A** |
| `C15` | I به‌عنوان اطلاعات متقابل | `eq:ch3-I-mutual-info; eq:ch3-I-entropy` | توزیع مشترک Ω_c V_cd | `app:ch3-I-bounds-proof` | هویت MI و ۰≤I≤H(Ω) بررسی شد | **A** |
| `C16` | I در برابر BT با پردازش داده | `eq:ch3-I-vs-BT` | برای q=۱/K، Ω یکنواخت | `eq:app-ch3-I-BT-data-processing` | کران برنولی؛ صعود کران نه لزوماً خود I | **A** |
| `C17` | فانو و خطای بازشناسی برچسب مقصد | `eq:ch3-I-Fano` | مشاهده تصادفی C_ξ نه کل بردار | `app:ch3-I-bounds-proof` | کران پایین خطای تصمیم، نه upper BA | **A** |
| `D01` | Softmax با KL و تجزیه NLL | `eq:ch3-NLL-R` | بازنمایی مشترک؛ فاصله مرکز خود متناهی | `app:ch3-NLL-concavity` | تجزیه جبری و نقش مخرج رقیبان | **A** |
| `D02` | تقعر exp(-KL) و کران NLL–I | `eq:ch3-NLL-weighted-I-bound` | Softmax بدون وزن کلاس؛ امتیاز -KL | `app:ch3-NLL-concavity` | Jensen و تجزیه KL معتبر؛ افت کران ≠ افت NLL | **A** |
| `D03` | هموارسازی و نسخه پایدار KL | `eq:ch3-stable-smoothed-representations` | η∈(۰،۱)؛ مرکز از نمونه هموار | `app:ch3-stable-KL-normalization` | حفظ میانگین و کران KL | **A** |
| `D04` | نرمال‌سازی امتیاز، سقف Softmax | `eq:ch3-normalized-softmax-pmax` | z میانگین صفر؛ نرم ≤√K | `app:ch3-normalized-softmax-ceiling` | Jensen/کوشی؛ p*_K و L_K صحیح | **A** |
| `D05` | کران NLL نرمال و حالت دودویی | `eq:ch3-normalized-NLL-binary` | K=۲؛ σ_i>۰ برای برابری دقیق | `app:ch3-normalized-NLL-bound` | وابستگی برابری به نبود تساوی روشن است | **A** |
| `D06` | NLL پایدار با ε | `eq:ch3-stable-normalized-error-bound` | ε>۰ و فاصله KL متناهی | `app:ch3-stable-KL-normalization` | نرم ≤√K و قرارداد تساوی؛ C_K تیز منتقل نمی‌شود | **A** |
| `D07` | حاشیه JS و BA دقیق | `eq:ch3-js-margin-distance-difference` | قاعده تصمیم و قرارداد رفع تساوی | `app:ch3-margin-robustness` | علامت حاشیه و کران محافظه‌کارانه سازگار | **A** |
| `D08` | کران هندسی خطای تجربی JS | `eq:ch3-js-class-error-bound` | δ_c>۰؛ یک بازنمایی مشترک | `app:ch3-js-error-proof` | مثلث و مارکوف؛ ارجاع دقیق شد | **B** |
| `D09` | اثر جهت حرکت بر حاشیه JS | `eq:ch3-fixed-center-pairwise-js-margin` | مراکز ثابت و تصویر هیلبرتی واقعی | `app:ch3-margin-robustness` | تغییر R به‌تنهایی تغییر حاشیه را تعیین نمی‌کند | **A** |
| `D10` | میانگین/واریانس حاشیه KL | `eq:ch3-margin-mean; eq:ch3-margin-variance` | مراکز مثبت؛ بازنمایی مشترک | `app:ch3-margin-error` | میانگین=KL مراکز، واریانس کوادراتیک | **A** |
| `D11` | کران کانتلی و اجتماع رقبا | `eq:ch3-margin-error-bound` | حاشیه متناهی؛ μ≥۰ | `eq:app-ch3-margin-Cantelli-proof` | کران تجربی، نه تعمیم آماری | **A** |
| `D12` | کران چرنوف تجربی | `eq:ch3-margin-Chernoff` | حاشیه متناهی؛ t>۰ | `eq:app-ch3-margin-Chernoff-proof` | مارکوف روی exp(-tM) و union | **A** |
| `D13` | کف NLL و کران خطای تصمیم | `eq:ch3-normalized-NLL-error-lower-bound` | نرم امتیاز ≤√K | `app:ch3-normalized-NLL-bound` | خطای واقعی تحت قاعده و تساوی کنترل شد | **A** |
| `D14` | سیگموئید τ و حساسیت نرمال‌شده | `eq:ch3-js-sigmoid-loss; eq:ch3-js-sigmoid-derivative` | τ>۰؛ قرارداد σ(-m/τ) | `app:ch3-margin-robustness` | مشتق -sech²/(۴τ)، gτ≤۱؛ نه grad سیاست | **A** |
| `D15` | مشتق حاشیه نسبت به سیاست | `eq:ch3-margin-loss-policy-gradient` | سیاست مشتق‌پذیر؛ ماسک ثابت؛ ورودی مثبت | `app:ch3-margin-gradient` | قاعده زنجیره‌ای و تساوی رقیبان جدا شده | **A** |
| `D16` | کران پایداری تصمیم با متریک JS | `eq:ch3-margin-perturbation-bound` | بازنمایی مشترک؛ کران اختلال هر مرکز | `app:ch3-margin-robustness` | شرط کافی، نه ضرورت ناپایداری | **A** |
| `E01` | نگاشت پیشرو و یکتایی مرکز سیاست ثابت | `eq:ch3-center-forward-map; eq:ch3-policy-center-unique-walk` | D، β، λ، H و سیاست ثابت | `app:ch3-policy-center-markov` | امید یکتا؛ وارون یک‌به‌یک نیست | **A** |
| `E02` | مجموعه مراکز دست‌یافتنی | `eq:ch3-attainable-centers-data-explicit` | خانواده Z و داده ثابت | `app:ch3-inverse-policy-proof` | توصیف اشتراکی؛ شمول در جعبه ضروری | **A** |
| `E03` | مثال چهار‌نمونه‌ای عدم‌یکتایی مؤثر | `eq:app-ch3-effective-nonunique-example` | H=۱؛ دو ویژگی کامل؛ t=.۲۵ و .۷۵ | `app:ch3-inverse-policy-proof` | اعداد و مراکز دقیق بازحساب شدند | **A** |
| `E04` | کران بلمن min/max با حذف مبدأ | `eq:ch3-bellman-value-recursion` | H≥۱؛ مثبت‌بودن وزن؛ هر D_b≥۳ | `eq:app-ch3-feasibility-Bellman` | برهان استقرای معکوس؛ ۳۲ تست toy PASS | **B** |
| `E05` | کران دوطرفه مؤلفه مرکز | `eq:ch3-center-component-upper` | میانگین روی کلاس غیرتهی | `eq:app-ch3-feasibility-bound` | لازم؛ نه کافی؛ اشتراک سیاست ضروری | **A** |
| `E06` | حد پایین JSD از مختصه ناسازگار | `eq:ch3-center-component-upper` | JSD با log طبیعی؛ بازه دقیق | `eq:app-ch3-feasibility-js-lower` | clamp برنولی و Pinsker؛ ۱۲ toy PASS | **A** |
| `E07` | گواه عددی برای ۶ گراف پایه §۴.۱۱ | `sec:ch3-inverse-center-policy` | فقط H=۱۲ β=.۴ λ=.۸؛ هدف binary64 ثابت | `code/experiments/bayes_center_reachability_standard/REPORT_FA.md` | محدود به مورد ممیزی مستقل؛ ۴۸ حساسیت cert نیست | **A** |
| `E08` | اصلاح GN ژاکوبین مرکز | `eq:ch3-inverse-GN` | تابع مشتق‌پذیر و μ>۰ | `app:ch3-inverse-policy-proof` | حل رگولاریزه خطی، نه تضمین حل معکوس | **A** |
| `E09` | فیبر مراکز و پایداری I | `eq:ch3-fixed-center-fiber; eq:ch3-fixed-center-I` | قید دقیق مراکز برابر هدف دست‌یافتنی | `app:ch3-policy-center-projection` | هر معیار فقط مرکز ثابت؛ R/حاشیه الزاماً نه | **A** |
| `E10` | فرافکن و جهت کاهش مماسی | `eq:ch3-fixed-center-projection` | رتبه ثابت موضعی و grad غیرصفر | `app:ch3-policy-center-projection` | حفظ مرتبه اول؛ مسیر خمینه شرطی | **A** |
| `E11` | تصحیح غیرخطی و تلرانس مراکز | `eq:ch3-fixed-center-residual` | z* شاهد دست‌یافتنی، ε_C>۰ | `app:ch3-policy-center-projection` | روش پیشنهادی، حفظ دقیق از residual≤ε برنمی‌آید | **A** |
| `E12` | عمومیت بهبود BA در بهینه‌سازی دومرحله‌ای | `eq:ch3-two-stage-fixed-center-optimization` | فقط وجود فیبر؛ خانواده مشخص | `research_journal/2026-11-08-metrics-margin-fixed-centers.md` | فرضیه تجربی آزمایش‌نشده؛ در متن نیز تضمین نشده | **C** |
| `E13` | گواه دست‌نیافتنی برای همه H/داده‌ها | `eq:ch3-center-component-upper` | نیازمند محصورسازی مستقل هر تنظیم | `code/experiments/bayes_center_reachability_standard/REPORT_FA.md` | گسترش از شش گراف پایه به همه تنظیمات تأیید نشده و در متن ادعا نمی‌شود | **C** |
| `E14` | قاعده برون‌گراف سیاست node_class | `eq:ch3-policy-family-shapes` | بدون تابع شروع F، پیش‌بینی تازه بی‌تعریف است | `tex/chapter4.tex:1809ff` | صرف BA آموزش نمی‌تواند BA اعتبارسنجی باشد | **A** |

### موارد B — اصلاحات قابل دفاع
- **C04:** `R_{KL}^{legacy,clip}` در نسخه قبلی فهرست نمادها به معادله JSD ارجاع داشت؛ قبلاً صفحهٔ قرارداد clipping در `par:ch3-KL-legacy-clip` مشخص شد، و اکنون `R_{KL}` از `R_{JS},R_{sqrt(JS)}` در دو ردیف و دو مرجع صحیح جدا شده است. `R_{KL}` دقیق با flooring عددی یکی نیست.
- **D08:** ارجاع کلی کران هندسی JSD در حدود خط ۱۹۷۸ به بخش عمومی کران‌های خطا تغییر کرد و مستقیماً `app:ch3-js-error-proof` را هدف می‌گیرد. خود کران یا اثبات دست‌نخورده است.
- **E04:** پیش‌شرط‌های پوشش ویژگی‌ها، مثبت‌بودن وزن‌ها، H و β/λ در ابتدای متن اصلیِ کران‌های بلمن صریح‌تر شد؛ ارجاع به `eq:app-ch3-feasibility-Bellman` افزوده شد. هیچ علامت min/max یا محدودیتِ هسته تغییر نکرد.

### موارد C — سؤال علمی باز که نباید به قضیه ارتقا یابند
- **E12، `eq:ch3-two-stage-fixed-center-optimization`:** وجود یک شاهد هم‌مرکز به معنای وجود جهت بهبود یا موفقیت الگوریتم در همه داده‌ها نیست. مسئلهٔ تأثیر بهینه‌سازی با مراکز دقیقاً ثابت بر BA در یک خانواده سیاست مشخص، **UNVERIFIED / OPEN_RESEARCH** باقی می‌ماند. اثر: **مانع گزارش عملکرد تضمینی**، اما مانع معرفی روش مشروط به‌عنوان پیشنهاد نیست. اقدام آینده: پروتکل پژوهشی مستقل با قیود و گواه residual؛ در این مأموریت نه اجرا و نه تبدیل به ادعای قطعی.
- **E13، `eq:ch3-center-component-upper`:** تعمیم گواه عددی خارج‌شدن هدف از کران بلمن از شش گراف پایهٔ ممیزی‌شده به همه افق‌ها، داده‌ها یا تنظیمات **UNVERIFIED** است. اثر: نباید claim عمومی `CERTIFIED` ساخته شود. اقدام: فقط اجرای ممیزی و محصورسازی خطای مستقل برای هر تنظیم بیرون از این دامنه؛ نه تفسیر عدم همگرایی حلگر به‌عنوان ناممکن‌بودن. **متن موجود این تعمیم را ادعا نمی‌کند.**

**D — INVALID:** در نتایج اصلی بررسی‌شده مثال نقض قطعیِ حل‌نشده‌ای برای ادعای فعلی متن شناسایی نشد؛ این گزاره معادل تضمین فقدان هر خطا در همه بخش‌ها نیست.

## B — ماتریس انتقال دانش و پوشش ژورنال‌ها

فهرست زیر همهٔ ژورنال‌های موجود در شاخهٔ ویرایشی (۳۰ مورد، شامل فهرست README) را، همراه با **دو ژورنال R02 که فقط در SHAهای آرشیوی بودند** ثبت می‌کند. ردیف‌های `RESEARCH_ONLY` به‌طور صریح از انتقال جزئیات مدیریتی/آزمایشی به LaTeX مستثنا شده‌اند؛ `SUPERSEDED` هرگز به‌عنوان نظریه پذیرفته‌شده بازنشر نشده است. `PARTIAL` و `MISSING` برای یافته نظری معتبر ضروری در مجموعهٔ بررسی‌شده باقی نماندند؛ این وضعیت به شواهد محدودکننده وابسته است، نه ادعای اینکه تمام ایده‌های آینده باید در پایان‌نامه بیایند.

| ID | منبع نسخه‌دار | یافته / قلمرو | محل انتقال یا مرجع | وضعیت | تصمیم |
|---|---|---|---|---|---|
| J01 | `research_journal/2026-11-08-chapter3-appendix-audit.md` @ `7325ea1c3f4a` | مدل/پیوست و تناقض‌های پیشین | `sec:ch3-interwoven; app:ch3-proofs` | TRANSFERRED | قراردادهای معتبر موجود؛ خطاهای تاریخی احیا نشد |
| J02 | `research_journal/2026-11-08-metrics-margin-fixed-centers.md` @ `272242c57f71` | R، I، حاشیه، زیان و فیبر ثابت | `sec:ch3-normalized-margin; sec:ch3-inverse-center-policy` | TRANSFERRED | محتوای نظری منتقل؛ آزمایش‌های پیشنهادی RESEARCH_ONLY |
| J03 | `research_journal/2026-11-09-ch3-first-600-audit.md` @ `bd1d229b9786` | بازبینی ابتدای فصل/فرض‌ها | `subsec:ch3-definitions` | TRANSFERRED | تعریف‌ها و شرط پوشش ویژگی تثبیت |
| J04 | `research_journal/2026-11-09-ch3-theory-to-experiments-handoff.md` @ `8629baf35456` | رابطه نظری با آزمایش‌های آتی | `sec:ch3-inverse-center-policy` | TRANSFERRED | گزاره‌های نظری درج؛ برنامه اجرا در پژوهش |
| J05 | `research_journal/2026-11-09-ch4-benchmark-standard-and-objectives.md` @ `f823c47274dc` | Benchmark و قرارداد هدف‌ها | `tex/chapter4.tex:1602ff` | RESEARCH_ONLY | فقط محدودیت عدم یکنواختی I/BA در متن اصلی |
| J06 | `research_journal/2026-11-09-ch4-experiments-inventory-and-plan.md` @ `883efff3f6c9` | موجودی آزمایش‌های فصل ۴ | `tex/chapter4.tex` | RESEARCH_ONLY | جزئیات اجرایی نباید وارد فصل ۳ شوند |
| J07 | `research_journal/2026-11-09-ch4-foundations-p0-p2-audit.md` @ `b7679d5bd2d4` | ممیزی D/E پایه | `tex/chapter4.tex:14ff` | RESEARCH_ONLY | قرارداد داده مولد به فصل ۴ اختصاص دارد |
| J08 | `research_journal/2026-11-09-ch4-unified-snapshot-pipeline.md` @ `43c2ecb4fa3d` | گردش داده و snapshot | `کد/شواهد فصل ۴` | RESEARCH_ONLY | به متن نظری منتقل نشود |
| J09 | `research_journal/2026-11-09-chapter3-theory-and-markov-handoff.md` @ `5f5e23ee07d7` | زنجیره مارکوف/بازآغاز/انحصار مبدأ | `app:ch3-policy-center-markov; app:ch3-policy-center-sequential-restart` | TRANSFERRED | قضیه و مثال نقض در فصل/پیوست موجود |
| J10 | `research_journal/2026-11-10-E-CH4-09-depth-effect.md` @ `ba0943c244f6` | اثر عمق گشت | `eq:ch3-capped-alpha; tex/chapter4.tex:1381ff` | TRANSFERRED | فرمول نظری و محدودیت شواهد تفکیک |
| J11 | `research_journal/2026-11-10-FIX-CH4-09-mc-exact.md` @ `9f6335ee33c8` | MC در برابر exact | `eq:ch3-matrix-forward-recursion; tex/chapter4.tex:1338ff` | TRANSFERRED | هم‌ارزی نظری ≠ تخمین محدود MC |
| J12 | `research_journal/2026-11-10-N-CH4-01-narrative.md` @ `017efa27509c` | نثر گزارش آزمایش | `tex/chapter4.tex` | RESEARCH_ONLY | تاریخچه اجرایی وارد فصل ۳ نشود |
| J13 | `research_journal/2026-11-10-ch4-45-46-report-integration.md` @ `6319e3fb9755` | گزارش هدایت‌های ۴.۵/۴.۶ | `tex/chapter4.tex` | RESEARCH_ONLY | جزئیات نه قضیه عمومی |
| J14 | `research_journal/2026-11-10-ch4-common-metric-contract.md` @ `de1d1b5abeef` | قرارداد پنج‌گانه سنجه‌ها | `eq:ch3-R-JS-forms; eq:ch3-I-on-V` | TRANSFERRED | KL دقیق/clip و معیارهای نمونه‌ای تفکیک |
| J15 | `research_journal/2026-11-10-ch4-common-metric-coverage-audit.md` @ `0e1923060a95` | پوشش اندازه‌گیری سنجه‌ها | `tex/chapter4.tex:1602ff` | RESEARCH_ONLY | پذیرش مرکزمحور به همه BA/NLL تعمیم ندارد |
| J16 | `research_journal/2026-11-10-ch4-e06-e07-benchmark-audit.md` @ `83a101654c88` | ممیزی benchmark | `tex/chapter4.tex:1602ff` | RESEARCH_ONLY | تفکیک گروه‌های وابسته و سه seed |
| J17 | `research_journal/2026-11-10-ch4-guided-probability-mixture-correction.md` @ `59bf9247a0d1` | اصلاح هدایت احتمال‌ها | `tex/chapter4.tex:644ff` | SUPERSEDED | نسخه نهایی و اصلاح‌شده در متن فصل ۴ لحاظ شده |
| J18 | `research_journal/2026-11-10-ch4-latex-compile-errors.md` @ `cd848e635859` | خطاهای کامپایل LaTeX گذشته | `گیت فنی جدید` | RESEARCH_ONLY | هیچ PASS چاپی بدون اجرا ادعا نشود |
| J19 | `research_journal/2026-11-10-ch4-one-step-q-report-standard.md` @ `a4b1d53ab733` | بازنمایی یک‌گامی Q | `eq:ch3-feature-one-step-Q` | TRANSFERRED | Q از امید چندگامی جدا باقی ماند |
| J20 | `research_journal/2026-11-10-ch4-preferred-experiment-sequence-proposal.md` @ `6ce327ce995d` | برنامه آزمایش‌های آینده | `roadmap/شواهد` | RESEARCH_ONLY | طرح تجربی، قضیه نیست |
| J21 | `research_journal/2026-11-10-ch4-presentation-unification-43-47.md` @ `98bfc0187109` | یکدستی نحوه ارائه نتایج | `tex/chapter4.tex` | RESEARCH_ONLY | نسخه چاپی یکدست؛ تاریخچه جدا |
| J22 | `research_journal/2026-11-10-ch4-sigmoid-margin-e07.md` @ `247bfd719ba0` | زیان سیگموئیدی | `eq:ch3-js-sigmoid-loss` | TRANSFERRED | قرارداد τ در مخرج درست |
| J23 | `research_journal/2026-11-10-ch4-uniform-sensitivity-report.md` @ `aa636d96dcb6` | سیاست یکنواخت و حساسیت | `eq:ch3-uniform-policy; eq:ch3-effective-depth` | TRANSFERRED | بدون تعمیم روند عددی به قضیه |
| J24 | `research_journal/2026-11-10-e08-execution-and-delivery.md` @ `541eb1ed5a54` | اجرای E08 و محدودیت‌ها | `eq:ch3-js-sigmoid-loss` | TRANSFERRED | محاسبه تجربی جداگانه؛ بدون تضمین BA |
| J25 | `research_journal/2026-11-10-e08-preregistration.md` @ `fc85b3d27c65` | پیش‌ثبت E08 | `پوشه آزمایش/فصل ۴` | RESEARCH_ONLY | جزئیات برنامه پژوهش |
| J26 | `research_journal/2026-11-11-ch4-research-master-inventory.md` @ `f47bb44921f4` | دفتر مادر پژوهش | `research_journal و Issue #49` | RESEARCH_ONLY | فهرست منشأ، نه روایت نظری |
| J27 | `research_journal/2026-11-11-ch4-research-master-roadmap.md` @ `57224bc76878` | نقشه راه پژوهش | `roadmap` | RESEARCH_ONLY | موارد ناتمام به قضیه تبدیل نشد |
| J28 | `research_journal/2026-11-14-ch3-final-editorial-qa.md` @ `f486398689ad` | ممیزی ویراستاری قبلی | `tex/chapter3.tex; tex/appendix_chapter3.tex; tex/notations.tex` | TRANSFERRED | اصلاحات قبلی محفوظ؛ PDF هنوز NOT_RUN |
| J29 | `research_journal/2026-11-14-ch3-postch4-theory-alignment.md` @ `a79c612c21e0` | تطبیق نظری پس از فصل ۴ | `eq:ch3-bellman-value-recursion; eq:ch3-fixed-center-fiber` | TRANSFERRED | اثبات بلمن دوطرفه، عدم‌یکتایی مؤثر، مرز گواه |
| J30 | `research_journal/README_FA.md` @ `14ba708cf119` | فهرست موضوعی ژورنال‌ها | `research_journal/README_FA.md; Issue #54` | RESEARCH_ONLY | برای کشف منشأ و گیت استفاده شد |
| J31 | `research_journal/2026-11-14-sec411-bayes-posterior-vs-neighborhood-reachability.md` @ `4ec413ebbc1e` (R02 unmerged) | پسین بیزی ≠ برچسب مقصد و وابستگی به D | `eq:ch3-attainable-centers-data-explicit` | TRANSFERRED | اصل تعریفی و عدم استلزام وجودی منتقل؛ سند آرشیوی جداست |
| J32 | `research_journal/2026-11-14-ch3-bellman-two-sided-proof-and-feasibility-bridge.md` @ `3d233de4cfe0` (R02 unmerged) | کران لازم دوطرفه با هسته مبدأحذف | `eq:ch3-bellman-value-recursion; eq:app-ch3-feasibility-Bellman` | TRANSFERRED | برهان نهایی و نماد U اصلاح‌شده؛ typo u_{s,u} از ژورنال عیناً منتقل نشد |

### ترتیب اولویت و اسناد ابطال‌شده
- **Bellman قدیمی با بازگشت مجاز به مبدأ** که در تاریخچه D10/D11 ثبت شده بود، برای مدل بدون بازگشت به مبدأ نامعتبر/منسوخ است. استناد نظری فقط به هستهٔ `q_b^{(s)}` و برهان دوطرفهٔ جدید `eq:app-ch3-feasibility-Bellman` است. ژورنال قدیمی، بدون تبدیل به نتیجه جدید، حفظ می‌شود.
- **R02 unmerged:** سند اثبات اولیه `3d233de4...` از نماد موقت `u_{s,u}` برای upper استفاده می‌کرد؛ LaTeX نهایی از `U_{s,u}` و `\overline U_{c,u}` استفاده می‌کند و محتوا را با نگارش و فرض‌های اصلی سازگار کرده است. جدیدتر بودن یک سند به‌تنهایی گیت علمی نیست.
- **اهداف بیزی** پسین برچسب خود نمونه هستند، نه برچسب مقصدهای گشت. `∃D` مستلزم `∀D` نیست، اما این استنتاج منطقی **به‌تنهایی مثال نقض وجودی برای ∀D نمی‌سازد**. محصورسازی عددی و مورد خاص مستقل از این گزاره‌اند.
- **شاخه‌های پژوهشی و مطالعات معکوس:** ۱۰۸ اجرای §۴.۱۰ در سه بذر مستقل داده، موفقیت تا آستانهٔ ازپیش‌تعریف‌شده را ثبت می‌کنند؛ این ۱۰۸ مشاهدهٔ مستقل و نه اثبات برتری عمومی GN هستند.
- **§۴.۱۲:** ۷۶ اجرای استاندارد و ۸۳۶ سنجهٔ پذیرفته‌شدهٔ **صرفاً مرکزمحور** وجود دارد؛ این پذیرش را به NLL/BA و سنجه‌های نمونه‌محور تعمیم نمی‌دهیم.

## C — فهرست خطاها و اصلاحات ثبت‌شده

| محل | ایراد | اصلاح | نوع | کامیت |
|---|---|---|---|---|
| `tex/chapter3.tex`، آغاز و گذار بخش‌ها | ابهام نثر، اصطلاحات نامنسجم | بازنویسی محدود و یکدست‌سازی | نگارشی | `583109e8`, `5f5c2dfa` (مرحله قبل) |
| `tex/appendix_chapter3.tex`، ضرایب/بلمن/مثال مؤثر | روایت دشوار؛ خطر تلقی تعمیم همگانی | شفاف‌سازی نثر و دامنه بدون تغییر فرمول | نگارشی-علمی | `3f86ae3e` (مرحله قبل) |
| `tex/chapter3.tex`، قرارداد clipping و مجموعه مراکز | ارجاع معنایی فاقد رابطه دقیق | برچسب پاراگراف و دو تعریف شماره‌دار اصلی | فنی | `5099d2e5` (مرحله قبل) |
| `tex/notations.tex` | `α_t^(H)` تکراری، `r_C` با هدف اشتباه | حذف تکرار و ارجاع دقیق | فنی | `0b9e9fe4` (مرحله قبل) |
| `tex/chapter3.tex`، حدود خط ۱۹۷۸ و ۴۲۱۴–۴۲۶۰ (label ثابت) | اشاره کلی به اثبات هندسی؛ پیش‌شرط ضمنی بلمن | مقصد اثبات مشخص؛ بیان صریح H,β,λ,|D_b| و ارجاع به بازگشت استقرایی | علمی-توضیحی | `c6f8e74db499d6a31f1ac1164d26225f71974b77` |
| `tex/notations.tex`، بخش معیارها/NLL/کران/حل معکوس | ردیف مشترک R با مقصد نادقیق؛ نمادهای کلیدیِ L_K،p*_K،η،ε،G_Ch،Π_J،γ_R غایب | تفکیک R و افزودن نمادها با label معرفی‌کننده | نمادگذاری | `479c34c03c5310de0df6c30ed361e3a1f228b061` |

**تغییر در قضیه/فرمول پذیرفته‌شده:** هیچ فرمول موجود جایگزین نشد. در مقایسه ماشینی با `main@7829e15`، ۲۰۴ بلوک ریاضی شماره‌دار فصل و ۶۹ بلوک پیوست (جمعاً ۲۷۳) به‌صورت متن کامل برابر مانده‌اند؛ دو بلوک شماره‌دار فصل صرفاً تعریف‌های نثری قبلی هستند. در مأموریت حاضر هیچ آزمایش آموزشی اجرا نشد.

## D — مسائل باز علمی و محدودیت رأی
۱. دو پرسش C بالا برای ادعاهای **گسترده‌تر از متن فعلی** بازند؛ هیچ نتیجهٔ شرطی به حقیقت همگانی تبدیل نشده است.  
۲. ناوردایی logits در Softmax اثبات عدم‌یکتایی **پارامتر** است؛ وجودی بودن عدم‌یکتایی **مؤثر** تنها با مثال چهار نمونه/H=۱ ثابت شده، نه برای همه گراف‌ها.  
۳. شواهد §۴.۱۱ شش گراف پایهٔ پذیرفته‌شده را پوشش می‌دهند؛ برنامه حساسیت‌های عددی/تکرار آزمایش‌ها را نباید برهان ریاضی یا گواه همگانی نامید.  
۴. اعلام کفایت همهٔ استنادهای دانشگاهی به ادبیات، صرفاً از سالم‌بودن کلیدهای `cite` استنتاج نمی‌شود. `BIB_METADATA_INDEPENDENT_EXTERNAL_CHECK=NOT_RUN`. بررسی کامل صفحهٔ نهایی `NOT_RUN`.  
۵. ممیزی خط‌به‌خط رسمیِ همهٔ گزاره‌های فرعی توسط بازبین مستقل دوم جایگزین این خودممیزی نمی‌شود. این گزارش ماتریس ادعاهای **مهم** را پوشش می‌دهد، نه پروانهٔ صحت بدون قید برای هر عبارت نثری.

## E — کنترل نماد، ارجاع و قیاس با فصل چهارم

- `\alpha_t^{(H)}` یک ردیف دارد؛ `\ell_τ=σ(-m/τ)`، `g_τ=4τ|\ell'_τ|` و $`beta/lambda/H` مستقل‌اند؛ `ξ` اندیس تصادفی است.
- `R_{KL}` دقیق به `eq:ch3-R-average` و دو سنجهٔ JS به `eq:ch3-R-JS-forms` ارجاع می‌دهند؛ clipping تاریخی فقط مرجع صفحه‌ای دارد. `\varepsilon_C` تلرانس قید و `\varepsilon_r` جرم حذف‌شده، `\varepsilon` منظم‌ساز امتیاز هستند.
- ردیف‌های مقدماتی و واژگان `e_j` در فضای نمونه (بعد n) و یک‌داغِ برچسب (بعد K) تفکیک شدند. بازنویسی سراسری نماد پذیرفته‌شده برای رفع چندمعنایی انجام نشد.
- متن اصلی برای تعریف گشت، محاسبه دقیق، معیارها، زیان، کران‌های دوطرفه و قیدهای مسئله معکوس حاوی فرمول و تعبیر کافی است؛ پیوست نقش اثبات را دارد. صورت صریح بلمن در متن اصلی و استقرای کامل در پیوست موجود است.
- `chapter4.tex` و پرونده‌های شواهد §۴.۱۰، ۴.۱۱، ۴.۱۲ فقط خوانده شدند؛ هیچ ادعای عمومی درباره افزایش یکنواخت BA با I یا R، بازیابی قطعی همه هدف‌ها، یا تعمیم node_class از Train نوشته نشد. ۷۶ اجرا ≠ ۷۶ بذر مستقل؛ ۱۰۸ اجرا ≠ ۱۰۸ آزمایش مستقل؛ سه seed مستقل داده‌اند.

## F — آزمون‌ها؛ PASS، FAIL، NOT_RUN دقیقاً از اجرا

| آزمون / روش | نتیجه | حدود |
|---|---|---|
| مقایسه Git blob با مبدأ: ۲۷۳ بلوک ریاضی قدیمی بی‌تغییر، +۲ تعریف شماره‌دار | **PASS** | مقایسه متن و ساختار، نه برهان ریاضی |
| بازبینی استقرای Bellman دوطرفه با مبداحذف | **PASS (MATH REVIEW)** | با فرض‌های مثبت‌بودن وزن و H محدود؛ نه گواه عددی همه داده‌ها |
| ۳۲ بررسی عددی toy (H=۱..۴، چهار مبدأ، دو سیاست) برای L≤actual≤U و Σα=۱ | **PASS (TOY)** | چهار نمونه، دو ویژگی، β=.۴، λ=.۸ |
| ۱۲ بررسی toy کران Bernoulli JSD clamp و Pinsker | **PASS (TOY)** | شامل هدف‌های .۹۹، .۱ و .۴۵، نه محاسبه رسمی D10 |
| ۳۰۰ کنترل جبری toy (۶۰ ماتریس احتمال، K=۲،۳،۴،۵؛ seed قطعی 917): R+I، کران NLL-I، BT≥۱/K، R_JS≤R_KL/۲، نرمال‌بودن مرکز | **PASS (TOY)** | نمونه‌های مصنوعی ریاضی، نه آزمایش مدل/آموزش |
| اسکن ایستای label/ref/cite و فایل‌های تصویر در محدوده فصل و فایل‌های اصلی | **PASS (STATIC SCAN)** | ورودی‌های تولیدشده `\input` فصل چهارم نیازمند کامپایل کامل‌اند |
| XeLaTeX / bib / xindy با checkout کامل و فونت‌های موردنیاز | **NOT_RUN** | موتور در محیط هست، اما checkout کامل وابستگی‌ها و assets در آن محیط فراهم نبود |
| بازبینی PDF: عرض فرمول، font فارسی، ارجاعات صفحه‌ای | **NOT_RUN** | `VISUAL_QA_PENDING` |
| بازاجرای ۱۰۳/۱۰۸ آزمایش / ممیزی مستقل شهادت عددی | **NOT_RUN (OUT OF SCOPE)** | فقط شواهد مهرشده/گزارش‌های پذیرفته‌شده خوانده شدند |

**مرز تضمین:** شمار toy از اجرای مستقل همان نمونه‌های کوچک حاصل است؛ برهان عمومی تنها از استدلال نمادین پیوست است. PASS ایستا به معنای PASS چاپی نیست.

## G — رأی نهایی و شرط ادغام
- **صحت علمی شرطی:** ادعاهای اصلی بررسی‌شده تحت فرض‌های ذکرشده، A/B گرفته‌اند؛ مسئلهٔ گواه همگانی یا بهبود BA در دو مورد C باز و در متن نیز به‌صورت قطعی مطرح نیست. موارد D قطعیِ حل‌نشده شناسایی نشد.
- **کفایت اثبات:** هویت‌های اصلی، کران‌های بلمن، هیلبرت، JS، NLL و قضایای مارکوفی در پیوست اثبات متناظر دارند؛ ممیزی مستقل بیرونی جایگزین نشده است.
- **کفایت متن اصلی:** بخش روش/معادلات اصلی بدون مطالعه اثبات طولانی قابل‌فهم است؛ عبارات قابل اصلاح از لحاظ توضیح فرض بلمن و آدرس اثبات اصلاح شدند.
- **انتقال دانش:** ۳۰ سند root و ۲ سند R02 مستقل به‌طور مستند طبقه‌بندی شدند؛ یافتهٔ نظری ضروریِ شناخته‌شده بدون تصمیم انتقال نماند؛ آزمایش‌ها، شکست‌ها و شواهد منسوخ به‌جای قضیه منتقل نشدند.
- **یکپارچگی:** متن/پیوست/نمادها در کنترل ایستا و مفهومی با شواهد فصل ۴ هماهنگ‌اند، اما **قبولی فنی PDF هنوز احراز نشده است**.
- **گیت ادغام:** `READY_FOR_MANAGER_SCIENTIFIC_REVIEW` **با `VISUAL_QA_PENDING`؛ مدیر باید کامپایل و PDF را ببیند، موارد C را در محدوده گزارش نگه دارد و گیت مستقل علمی را بررسی کند**. تا آن زمان **DO NOT MERGE**.

آخرین SHA شاخه و لینک PR در metadata زندهٔ GitHub نگهداری می‌شوند؛ SHA نهایی commit ایجاد همین گزارش داخل فایل خودش قابل درج به‌صورت self-reference نیست.
