# RESEARCHER-03 — P2 minimal decision (DESIGN ONLY; NOT AUTHORIZED)

**Mission:** `CH4-METRICS-P3-PUBLICATION-GATES-01` • **Role:** RESEARCHER-03 • **Baseline:** `P1@d36ec6dc522e4e6d5f7ee4134f23c71f97b2aff3` • **Scientific source:** `chapter4@de02fdda91e23db177de4429b3e1a9693c046d83` • **Status:** `PROPOSAL_NOT_EXECUTED`.

## A. اصل کم‌هزینه بودن: نبود عدد الزاماً نیازمند inference جدید نیست

1. **Do nothing (بدون forward):** مرکزهای موجود، تعریف وضعیت `not_applicable`، اغلب aggregateهای تاریخی و دو metadata اجرا را به‌عنوان شواهد نسخه‌دار حفظ کنید. `R_JS`, `R_sqrt_JS`, BA و confusion تاریخی با گواه جبر aggregate قابل نقل مشروط‌اند. **برای ادعای `verified_v2` صرف وجود عدد P1 کافی نیست.**
2. **Archive inventory first:** قبل از حتی اجرای P2، فهرست ذخیره‌های موجود و `backfill_saved_scores.py --per-sample` احتمالی بررسی شود. `posthoc_score_backfill_76_compact.csv` صرفاً شش رقم و خلاصه‌ها را دارد؛ `raw_logits` در checkpointهای اصلی موجود است، اما P1 `P_i^{(c)}` / `S_{i,c}` را ثبت نکرده است. اگر قبلاً per-sample output با SHA معتبر پیدا شد، فقط **خواندن** و اعتبارسنجی آن، ارزان‌تر از forward است؛ استفاده بدون تأیید هویت داده/سیاست ممنوع.
3. **Micro-pilot after independent QA76 and separate approval only:** ابتدا فقط دو checkpoint از `shared/class`، سپس گیت تصمیم؛ در صورت نیاز به پوشش Train خانواده گره‌ای، دو checkpoint `node/node_class` به‌عنوان یک مرحلهٔ افزوده و **صرفاً Train**. حتی پایلوت چهارگانه، تأیید ۷۶ اجرا نیست.

## B. پایلوت پیش‌ثبت‌شده با حداقل تغییر هم‌زمان

**قاعده انتخاب از پیش قفل:** `data_seed=20261102`، `n_ref=50`، `objective=margin_JSD`، `training_tau=0.005`، `steps=25`، دقیقاً یک عضو از هر خانواده. **این انتخاب به BA یا Test نگاه نمی‌کند** و حجم کوچک گراف را هدف می‌گیرد؛ همه در آرشیو `size_25_24runs_seed20261102.zip` با **SHA256 کامل** `d570fb3f1dcab66476f027e1ade17b4e87c7ca16ff7fa6c76fda87b260c84e9f` ثبت هستند.

| مرحله | خانواده | نام دقیق عضو JSON | member SHA-256 | Train | Validation |
|---|---|---|---|---|---|
| **A (کمترین پایلوت)** | `shared` | `ds20261102_n50_shared_margin_JSD_tau0.005_s25.json` | `0704839ef1bf4d930a90c7a6111341533fc4218d435b984e2d4045683b78cfbb` | مجاز در P2 آینده | مجاز در P2 آینده |
| **A** | `class` | `ds20261102_n50_class_margin_JSD_tau0.005_s25.json` | `0538105ba68dd1e5d69e30e8beb205f85f9b2e1376c1f498b2e3da152c3a7581` | مجاز در P2 آینده | مجاز در P2 آینده |
| **B فقط در صورت لزوم** | `node` | `ds20261102_n50_node_margin_JSD_tau0.005_s25.json` | `891277766219620ea28e26b757797705f0c1b0ea024758d188bef01e5c4448e2` | فقط Train | **N/A بدون query-start** |
| **B فقط در صورت لزوم** | `node_class` | `ds20261102_n50_node_class_margin_JSD_tau0.005_s25.json` | `bd1bd205eef3ee180c91434d0bd70e3712ff10204a2fbcec0bae640ef5911bea` | فقط Train | **N/A بدون query-start** |

**گیت اجرا:** ۱) رأی مستقل `ACCEPT_QA76_CENTER_ONLY`؛ ۲) پذیرش رسمی و مجوز **جدید** P2 با قفل چهار member، SHA ZIP، `master_runs_76.csv@bcc2d7f4...`، `code/data_generator/config.json@047d3758...`، `engine.py@9b1cd83...`، `backfill_saved_scores.py@3ace90be...`، `report_snapshot.py@26a3c2f...` و `COMMON_METRICS_CONTRACT.json@a7e1cea6...`؛ ۳) کنترل منشأ Train/Validation، reference IDها و ترتیب/برچسب‌ها بدون ساخت Test؛ ۴) گذر از حفاظت `masked_softmax`، exact KL، tie و checkpoint consistency. **اولین mismatch = STOP، بدون جایگزین کردن scalar تاریخی با مقدار جدید.**

## C. حداقل داده لازم برای معیارهای فاقد گواه v2

| هدف/شکاف | داده تک‌نمونه‌ای ضروری | محاسبهٔ مجازِ پیشنهادشده برای آینده | ریسک/برچسب |
|---|---|---|---|
| `R_KL` exact و `R_KL_exact_support_aware` | `P_i^{(y_i)}`، (y_i)، `V_y` و ماسک support | `D_KL(P||V)` بدون floor و ثبت `+∞` برای support mismatch، class-balanced | خود `R_KL_legacy_clipped` شاهد عدد دقیق نیست |
| `KL_candidate_support_zero_count` | **همه** `P_i^{(c)}` و `V_c` | ارزیابی support برای تمام فرضیه‌های قابل محاسبه، ثبت convention شمارش | positivity center تنها یک **گواه نظری** است، نه شمارش empirical |
| `exact_tie_count`, `near_tie_count_atol_1e-12`، `tie_count` | همه `S_{i,c}`، `argmax` و label برای خطا | برابری عددی دقیق جدا از `np.isclose(...,atol=1e-12,rtol=0)`; argmax در تساوی کمترین اندیس | نام تاریخی `tie_count` الزاماً دقیق نیست |
| `margin_quantiles`, hard margin و sigmoid | `S_{i,c}`، (y_i)، `tau_report` با نسخه | margin true-minus-largest-rival و میانگین هم‌وزن کلاس؛ فقط برای **verify-v2** | Quantile aggregate تاریخی برای بازسازی همه margins کافی نیست |
| `balanced_JS_similarity_NLL`, `balanced_softplus_all_rivals` | `S_{i,c}`, (y_i)، `tau_report` | NLL بر پایه softmax شباهت JS؛ softplus همه رقبا و **نسخه‌ی τ** | JS score احتمال بیزی کالیبره نیست؛ `tau=.005` posthoc ممکن است با training τ فرق داشته باشد |
| `NLL_KL_simple`, `NLL_KL_normalized_historical` | همه `P_i^{(c)}`, `V_c`، (y_i) | KL هر نامزد و normalization نسخه‌دار؛ پرهیز از ترکیب بی‌نام clipped/strict | `NLL_KL_normalized_historical` همان eta/epsilon جدید فصل ۳ نیست |
| `R_JS`, `R_sqrt_JS`, BA و confusion | برای **محاسبه از صفر/تأیید واقعی** `P_i^{(c)}`, `S_{i,c}`, (y_i) | فقط در صورت نیاز به ارتقای `historical` به `verified_v2` | جبر per-class/ماتریس آشفتگی P1، forward نیست |
| `center.*`، `optimizer_objective_loss`, `elapsed_s` | **هیچ خروجی P2 لازم نیست** | QA مستقل raw مراکز؛ metadata تاریخی حفظ شود | P2 برای این شاخص‌ها خرج بی‌مورد است |

در هر P2، **مراکز Train از checkpoint قفل‌شده ثابت می‌مانند**؛ هیچ مرکز از Validation به‌روزرسانی نمی‌شود و برچسب Validation فقط پس از تولید scoreها برای سنجش خطا استفاده می‌شود. `shared/class` باید **همه کلاس‌های فرضی** را برای query ارزیابی کنند. خانواده‌های `node/node_class` آزاد در Validation حتی با این پایلوت نیز `not_applicable` می‌مانند. سیاست `class_tied_node` در آزمایش دیگر، قرارداد شروع جدید برای این ۲۶ record نیست.

## D. پیشنهاد بودجه/سنجش و معیار توقف — **تخمین غیرمصوب**

- **مرحله صفر، inventory:** صرف خواندن فهرست artifactهای per-sample احتمالی؛ هدف زیر ۱ دقیقه و RAM کمتر از ۲۰۰ MiB، تخمین نه benchmark.
- **مرحله A، دو checkpoint:** تخمین اولیهٔ محافظه‌کارانهٔ **۶۰–۱۸۰ ثانیه** wall عددی و **RSS≤۱ GiB**؛ زمان واقعی در این مأموریت اندازه‌گیری نشده. از `torch.no_grad`، thread محدود و streaming خروجی استفاده شود. **قفل/توقف فیزیکی** در تجاوز زمان/RSS یا هر mismatch. برای دو run و (n_{mathrm{Train}}=50)، Validation هر دو در حدود ۹۰ نمونه مطابق داده تاریخی است؛ داده‌های لازم (P(n,K,K)) و (S(n,K)) کوچک‌اند. برآورد خروجی فشرده `<2 MiB` (با metadata، نه تضمین).
- **مرحله B (اگر لازم و جداگانه تصویب شد):** دو node-family با فقط Train، هدف budget مشابه مرحله A و برآورد خروجی `<1 MiB`. اجرای B نباید به‌عنوان توجیه سرخودِ Validation node تعبیر شود.
- **مرحله C، پوشش ۷۶ اجرا:** **در این مأموریت پیشنهاد اجرا یا budget مصوب ندارد**. تنها اگر بعد از A/B روشن شد برای ادعاهای مشخص واقعاً لازم است، درخواست جدید برای محدودهٔ موردنیاز، زمان‌بندی و واگرایی‌های نسخه‌ای ارائه شود. از pilot دو/چهارتایی نمی‌توان نتایج عددی ۷۶ اجرا را استنتاج کرد.

**فیلدهای خروجی پیشنهادی:** `run_id/family/seed/ref_ids/archive_member_sha/data_sha/center_sha/engine_sha/evaluator_sha/split`، shape/finite/support flags، `P_by_hypothesis`، `S_all_candidates`، (y)، pred، margin، tie_exact/near_tie، KL flags، metric statuses/version، `run_budget.json`، `comparison_against_saved_aggregate.json`. *فقط Train/eligible Validation؛ بدون Test.*

## E. Go / No-go

**GO** فقط با رأی QA76 کامل، مجوز جدید، SHAهای صحیح، fingerprint داده و تحقق بودجه محافظه‌کارانه صادر می‌شود. **NO-GO** در صورت `ACCEPT_WITH_LIMITATIONS` مستمر، مغایرت source/centers، نداشتن starts، استفاده از label واقعی query در policy، مختلط‌شدن `clipped/exact`، یا Test leakage.

**نتیجه P3:** اجرای P2 اکنون **ممنوع** است؛ صرفاً طراحی مرحله A/B و شرط پذیرش آماده شده است. — **RESEARCHER-03**
