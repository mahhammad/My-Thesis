# RESEARCHER-06_READINESS_REPORT — E-CH4-13
**۱۴ نوامبر ۲۰۲۶ | پژوهشگر:** RESEARCHER-06 | **مدیر:** RESEARCH-MANAGER-04
**وضعیت:** READY_FOR_PREFLIGHT_REVIEW / DESIGN_ONLY / NO_HEAVY_RUN
**مبنا:** main@ac5892ad166ce300a2044e3303d4adaa1a21d7d7 (پس از PR #77) | Issue #78
**دامنه:** مقایسه کنترل‌شده تابع زیان IMC؛ آموزش جدید، Test و تغییر LaTeX انجام نشده است.

## ۱. مدل و استنتاج
در IMC حالت هر گشت زوج (نمونه،ویژگی) است؛ گام کامل از انتخاب ویژگی بعدی (self-loop ویژگی در صورت مجاز بودن) و جهش عمودی به نمونه دیگر تشکیل می‌شود. ماسک ویژگی ناموجود، یال غیرمجاز و ممنوعیت بازگشت به *مبدأ همان قطعه گشت* در کل طول گشت اعمال می‌شوند؛ بازدید مجدد دیگر نمونه‌ها ممکن است. طول گشت هندسی با β است، λ وزن گام‌های مشاهده‌شده را کنترل می‌کند، H سقف محاسبه دقیق است؛ تقریب ماتریسی با MC یکسان نیست. قرارداد پایه β=.4، λ=.8، H=12، dtype=float64 است.

چهار خانواده عددی Z: shared=(d+1,d)، class=(K,d+1,d)، node=(n,d+1,d)، node_class=(K,n,d+1,d). سطر اضافی مربوط به *شروع* است؛ class به فرضیه کلاس مبدأ وابسته است نه برچسب گره میانی. پیش‌بینی برای نامزد c از بازنمایی مخصوص فرض c و مرکز c استفاده می‌کند. مراکز V_c صرفاً میانگین بازنمایی‌های **Train کلاس c** هستند و در Validation ثابت می‌مانند. برای هر نمونه Validation همه K فرضیه بدون y_val اجرا می‌شوند؛ برچسب Validation فقط پس از تولید logits/predictions برای سنجش به کار می‌رود. تنها shared/class بدون قانون شروع افزوده، بیرون‌گراف معتبرند. node/node_class آزاد به query_start قابل‌اثبات و مجزا نیاز دارند؛ Validation بدون آن **undefined**, نه صفر. مدل ترکیبی class_tied_node نیز خانواده مستقلی است و در مطالعه اصلی وارد نمی‌شود.

با امتیاز تصمیم s_ic=1-JSD(Y_i^(c),V_c)/ln(2)، پیش‌بینی argmax با رفع تساوی به کمترین اندیس و حاشیه واقعی m_i=s_i,y−max_(r≠y)s_i,r تعریف می‌شود. BA میانگین recall کلاس‌هاست، نه Accuracy معمولی. R_KL=D_KL(Y_i||V_y) جهت‌دار و support-aware، R_JS میانگین JSD و R_sqrt_JS **میانگین جذرهای تک‌نمونه** است. T/BT/TNL/BTNL/I از مراکز Train با وزن کلاس برابر محاسبه می‌شوند. هیچ‌یک به تنهایی تضمین BA نیست.

تابع سیگموئیدی **پهنایی** ℓτ(m)=σ(−m/τ) و مشتق −ℓτ(1−ℓτ)/τ است؛ از فرم تاریخی σ(−τm) متمایز. در m=0 گرادیان نسبت به m برابر −1/(4τ) است و در tie، ℓτ=.5 ولی خطای محافظه‌کارانه ۱ است. مشتق max رقیب در tie غیر یکتا است. Softplus چندرقیبی با τ log∑exp((s_ic−s_i,y)/τ)، NLL روی JS، NLL ساده بر −KL، NLL نرمال تاریخی بر -(d−mean(d))/std(d)، و max_I / combined_I_R_margin تعریف‌های متفاوت و gradient scale متفاوت دارند؛ به‌صورت نادرست هم‌نام نشوند.

## ۲. شواهد موجود و شکاف واقعی
مرجع خام موجودی **master_runs_76.csv** با ۷۶ ردیف، data_seed, objective, family, tau, steps, file و sha256 است (blob: bcc2d7f4a4f39301973eb53c465ff05a6b4d0f21). جدول حاضر **شمار اجرا**ست، نه شمار seedهای مستقل:

| بلوک | خانواده | objective | seed مولد | بودجه | وضعیت استنباط |
|---|---|---|---|---|---|
| core / ۱۶ | ۴ خانواده × (max_I + margin_JSD با τ=.002/.005/.020) | 20261102 | n=300، ۵۰ گام Adam .12 | موجود؛ shared/class دارای Validation، node* فقط Train؛ تک‌مولد |
| extra / ۱۲ | ۴ خانواده × (margin_sqrtJSD، margin_KL، combined_I_R_margin) | 20261102 | n=300، ۲۵ گام Adam .12 | موجود؛ بین سه objective همین بلوک paired محتمل است، باید init/indices raw تصدیق شود؛ با core ۵۰گامی **هم‌بودجه نیست** |
| size / ۴۸ | ۴ خانواده × nهای ۵۰ تا ۳۰۰، ۲۵ گام margin_JSD τ=.005 (۲۴ پایه + ۲۴ الحاقی shared/class) | 20261102/03/04 فقط برای shared/class؛ سایر خانواده‌ها محدودتر | ۲۵ گام؛ زیرمجموعه‌های درون مولد تو در تو | ۳۶ اجرای shared/class سه‌بذری در شش n؛ **تنها یک objective**، نه مقایسه چندبذری lossها |
| gapfill / ۸ | shared/class × NLL_JS_similarity، NLL_KL_simple، NLL_KL_normalized، softplus_JS_tau005 | 20261102 | n=300، ۲۵ گام Adam .12، init صفر | آموزش مستقیم در GitHub؛ شاهد تک‌مولد و با توابع/مقیاس متفاوت |
| retrospective / ۷۶ | posthoc NLL/Softplus/sigmoid روی logits ذخیره‌شده | مطابق checkpoint | optimizer=۰ | بازامتیازدهی **نیست** آموزش با همان loss |
| E-CH4-07 | ۱۸ ارزیابی تشخیصی (۳ checkpoint × ۶ τ)، ۳ آموزش ۵۰گامی shared و ۳۶ size run | موجود | مختلف | ۱۸ تشخیص seed مستقل نیست و τ=.020 با .005 نتایج متفاوت دارد |

شاهد موجود: در داده seed=20261102 برای shared با ۲۵ گام، Validation BA سیگموئید قبلی τ=.005 برابر 97.96%، NLL_JS حدود 89.46%، NLL_KL_simple حدود 89.46%، NLL_KL_normalized حدود 97.28% و Softplus حدود 96.77% گزارش شده؛ در class، NLL_normalized حدود 98.64% ثبت شده است. این **نتیجه تک‌بذر تاریخی** و به معنی برتری پایدار نیست. در core ۵۰گامی، بیشینه I با BA پایین‌تر از حاشیه هم‌زمان شده و در checkpointهای مقید کاهش R لزوماً افزایش BA نداده است.

**شکاف واقعی:** هیچ جدول کامل loss×(shared/class)×سه مولد با seed/init/budget یکسان و raw تأییدشده برای تمام objectiveهای مورد نظر موجود نیست؛ اثر scale/τ و چندشروعی، اختلاف بین seeds و gradient conflict کامل اثبات نشده‌اند. تکرار ۷۶ run، ۸ اجرای NLL یا سه τ پیشین به‌تنهایی ارزش افزوده ندارد. برای pairing واقعی باید raw هر جفت را با manifest و reference IDs راستی‌آزمایی کرد؛ تا آن هنگام وضعیت cross-block=NOT_VERIFIED است.

منابع: benchmark_v1/RESULTS_FA.md، GAPFILL_2026-11-08_FA.md، results/master_runs_76.csv، results/new_objectives_25_summary.csv، sigmoid_margin_observational_evaluation/REPORT_FA.md، IDEA-006. PRهای قبلی QA76 تنها **center-only** ۸۳۶ مقدار را پوشش داده‌اند؛ QA تمامی BA/NLL/حاشیه‌های تاریخی blanket-accepted نیست.

## ۳. API و اصول reuse
- code/experiments/benchmark_v1/policy_interface.py: create_policy(family,parameters,*,n,k,d,n_query=0) و evaluate_policy(dataset,policy_family,policy_parameters,*,gamma,beta=.4,lam=.8,horizon=12). دومی آموزش/گزارش انجام نمی‌دهد.
- code/experiments/benchmark_v1/snapshot_pipeline.py: collect_representations(), save_snapshot(), load_snapshot(), array_fingerprint(), config_fingerprint(). schema=imc-snapshot-v2؛ دو فایل immutable model.json و distributions.npz؛ raw باید SHA256 داشته باشد.
- code/experiments/benchmark_v1/report_snapshot.py: standard_metrics(by_hypothesis,centers,labels,tau,*,split,per_sample=True)، report_snapshot()، save_report(). این **reporter اصلی قرارداد کلاس‌فرضی** است، نه metrics.report() با ورودی ساده N×K.
- code/policy_evaluator/metrics.py: jsd، sqrt_jsd، kl_divergence (support-aware)، center_metrics، class_centers، classify، report، normalized_nll، soft_margin_loss؛ center_metrics را reporter مشترک import می‌کند. برخی توابع مخصوص بازنمایی مشترک N×K اند و نباید جای امتیاز فرضیه‌ای N×K×K به کار روند.
- code/experiments/benchmark_v1/replay_v1.py: objective_outputs، objective_loss، fit، forward؛ آموزش sigmoid/max_I/margin_KL/margin_sqrtJSD/combined و اصطلاحات historical را فقط پس از test تطبیق API reuse می‌کنیم.
- code/experiments/benchmark_v1/train_missing_objectives.py: train_loss و load_graph/one_run برای چهار NLL/Softplus؛ ثابت بودن τ=.005 و init صفر در کد باید هنگام طراحی توسعه ثبت شود.
- code/experiments/benchmark_v1/engine.py: ExactMaskedWalk، گشت و موتور exact؛ کد مستقل جایگزین نوشته نمی‌شود.
- code/experiments/COMMON_METRICS_CONTRACT.json (v2)، COMMON_METRICS_FA.md، EXPERIMENT_REPORT_STANDARD_FA.md و REPORT_PRESENTATION_FA.md: نام فیلد، پوشش و تولید جدول از یک raw.

**ناسازگاری‌های بالقوه که باید تست شوند:** metrics.py:report با score یک بازنمایی در برابر تمام مراکز کار می‌کند؛ standard_metrics برای هر نامزد بازنمایی خودش را می‌گیرد. metrics.py:classify نیز تساوی را به کمترین اندیس می‌شکند ولی «exact tie» در reporter با atol=1e−12 گزارش می‌شود؛ تعریف tie_count عددی و argmax باید مستقل کنترل شود. تابع KL سنتی آموزش با clamp_min(1e−300) همان support-aware KL گزارش نیست. NLL_KL_normalized_historical همان smoothing نظری فصل ۳ نیست. در TRAIN میانگین مراکز، KL متناهی است اما در Validation ممکن است ∞ شود؛ مقدار null باید با coverage توضیح داده شود. رفتار derivative برای sqrt(JSD) نزدیک صفر و NLL normalization با std≈0 نیاز به fixture دارد.

## ۴. پیش‌ثبت مطالعه پیشنهادی (نه مجوز اجرا)
**واحد مقایسه** (generator seed, family, initialization seed, same reference IDs, same data split, objective). در طراحی اصلی فقط shared/class و n=300، K=3، d=5، ۹۰ Validation، β=.4 λ=.8 H=12، exact float64. داده‌های مستقل پیشنهادی 20261102/03/04 از مولد موجود؛ seed init جدا (برای paired zero-init مقدار شروع یکسان است و نیازی به ادعای «استقلال init seed» نیست)، RNG optimizer و هر nondeterminism جدا ثبت می‌شود. اگر random start پذیرفته شود، seed init از seed مولد جدا و در همه objectiveها جفت خواهد شد. ref IDs و train/val fingerprints، کلاس‌توازن، gamma، mask، architecture و کل ترتیب داده در هر pair یکسان باشند. عدم استفاده از Test اجباری.

**طرح اولیه candidateها**: ۱) sigmoid JSD τ=.005 مرجع، ۲) Softplus-JS τ=.005، ۳) NLL_KL_normalized_historical، ۴) NLL_KL_simple (چهارم در صورت تأیید هزینه/پیش‌پرواز). margin_KL / margin_sqrtJSD / max_I / combined و τهای .002/.020 **فقط توسعه بعدی مصوب**؛ فهرست قدیمی اجازه full-grid نیست. دو لایه نتیجه: مقایسه fixed optimizer steps/lr و تحلیل حساسیت scale/gradient norm؛ equal steps به معنای equal effective progress نیست. «equal compute» با بودجه ثابت steps/gradient evaluations و زمان/حافظه اندازه‌گیری‌شده، نه الزام برابری runtime. هیچ انتخاب پسینی بهترین τ بر اساس Test.

**تحلیل اصلی:** Validation BA paired و ΔBA برای هر مولد/خانواده، recallها و flip count، تغییر حاشیه صدک‌ها و hard error؛ secondary: Train R_KL دقیق و R_JS/R_sqrt_JS، I/T/BT/TNL/BTNL، gradient norm، initial/final training objective هرکدام در مقیاس خودش، زمان و حافظه. SD سه مولد فقط توصیفی و CI قطعی ادعا نشود. در تمام checkpointها loss/gradient و BA با همان تعریف محاسبه شود؛ گرادیان کلاس/نمونه برای تحلیل optional، نه برای آغاز پایلوت. ملاک توقف از قبل قفل شود؛ هیچ early stopping انتخابی از Validation در مقایسه اصلی.

**طرح multi-seed مشروط:** پس از پذیرش پایلوت، ۳ مولد × ۲ خانواده × ۳ objective اولیه × ۲۵ گام = ۱۸ fit؛ با اضافه‌شدن NLL_KL_simple ۲۴ fit. بازاستفاده از checkpointهای قدیمی فقط با شرط raw identity، optimizer config، code semantics و SHA معتبر؛ در غیر این صورت برچسب non-comparable. جدول های validation باید fold/seed-paired باشند، هر سه داده واحد مستقل، نه ۹۰ نمونه یا ۲۵ checkpoint مستقل.

## ۵. تست‌ها و STOP
| گیت | تست سبک و مرجع | وضعیت این checkpoint |
|---|---|---|
| موتور / mask / blocked / origin / exact / one-multistep | fixture graph ۲–۶ نمونه و مرجع ماتریسی مستقل، MC فقط tolerance آماری | NOT_RUN |
| گرادیان و زیان، tau<=0، near-tie، 0/∞/NaN، finite differences | replay_v1/train_missing_objectives و مرجع مشتق تک‌متغیره + autograd | NOT_RUN |
| KL دقیق vs legacy، JSD و mean(sqrtJSD) | metrics.py + hand fixtures و support | NOT_RUN |
| هندسه R/I/T/BT/TNL/BTNL و وزن کلاس | metrics.py:center_metrics و hand fixtures | NOT_RUN |
| BA/confusion/recall/tie/margin و scorer فرضیه‌ای | report_snapshot.standard_metrics + استقلال predictions از y_val | NOT_RUN |
| split/reference/seed و fingerprint/checksum/roundtrip | snapshot_pipeline.py + known raw sample | NOT_RUN |
| reporter vs raw/coverage/table/fig consistency | COMMON_METRICS_CONTRACT.json + بازساخت کامل | NOT_RUN |
| CI/QA independent/XeLaTeX/PDF | مرحله‌های بعدی | NOT_RUN |

**STOP hard:** نشت y_Validation یا هر Test؛ NaN/Inf تعریف‌نشده در training loss/gradient؛ شکست simplex/mask/origin؛ اختلاف checksum، raw/fingerprint/config؛ node/free query بدون شروع؛ عدم انطباق BA/margin/center با API؛ تعریف unsupported KL به‌عنوان finite؛ تناقض τ یا mismatch count. هر FAIL بلافاصله جلوی optimization را می‌گیرد و خطا/fixture باید حفظ شود. FAIL به NOT_RUN تبدیل نمی‌شود. برای نبود یک مقدار معتبر، coverage.json با not_applicable/not_measured/undefined_or_infinite/not_verified و علت ثبت شود؛ صفر واقعی measured است.

## ۶. بودجه پیشنهادی پایلوت — نیازمند مجوز مدیر
**فقط دو fit:** یک مولد 20261102، n=50 (زیرمجموعه طبقه‌متناسب ثابت و ثبت‌شده)، خانواده shared، دو objective sigmoid JSD τ=.005 در مقابل Softplus JS τ=.005، logits صفر جفت‌شده، Adam lr=.12، ۱۰ step برای هر fit، β=.4 λ=.8 H=12، float64؛ یک checkpoint ابتدا و آخر و checkpoint میانی در صورت کم‌هزینه بودن؛ ثبت gradient norm و trace، raw کامل و coverage. این دو هدف دارای واحد امتیاز یکسان و پارامتر τ یکسان‌اند ولی scale loss/gradient یکسان نیست؛ آن‌ها را «هم‌مقیاس» ننامیم. برای کنترل تطبیق، قبل از fit یک snapshot uniform بدون آموزش و expected exact را با reporter مشترک روی همان داده بسنجیم. فقط اگر همه تست‌های سبک PASS و مدیر گیت بدهد. پس از پایلوت، QA مستقل و تصمیم درباره grid سه‌بذری.

## ۷. ریسک‌ها و تصمیم‌های لازم
1. **مقیاس objective:** Adam یک lr مساوی به معنای اثر گرادیانی یکسان نیست. ثبت norms/gradient direction، عدم ادعای برتری ذاتى؛ normalization احتمالی تنها به‌عنوان مطالعه جدا با preregistration.
2. **مقیاس حاشیه:** JS similarity، sqrtJS و KL واحد یکسان ندارند؛ τ یکسان الزاماً sensitivity یکسان نمی‌سازد؛ بدون calibration پیش‌ثبت‌شده «مقایسه τ برابر» تفسیر علّی محدود دارد.
3. **non-smooth:** max rival در تساوی‌ها؛ finite diff باید نقاط دارای رقیب یکتا را انتخاب کند؛ tie fixture جدا.
4. **نشتی/شروع:** برچسب validation هرگز ورودی ساخت policy نیست؛ class candidate hypothesis فقط فرضیه است.
5. **رابطه R/I/BA:** بهبود هندسه بدون بهبود حاشیه ممکن؛ اثر هم‌زمان و مقایسه با نرخ خطا ثبت شود.
6. **نمونه آماری:** سه مولد تخمین اثر robust نیست؛ ترتیب رتبه‌ها ممکن است وارونه شود.
7. **provenance:** archived ZIP, raw JSON و manifest باید قبل از هر reuse بررسی شوند؛ تفاوت نسخه گزارشگر و training code علامت‌گذاری شود.
8. **کیفیت گیت:** هیچ PASS برای کد/CI بدون اجرا، هیچ پذیرش blanket برای ۵۵ سنجه و هیچ ادعای نتیجه جدید از طراحی.

## ۸. تحویل / جانشینی
**READ:** انجام شد: شش سند دانش جاری، فصل ۳ و پیوست و فصل ۴ (بخش‌های مربوط و ساختار کل)، قرارداد معیارها، APIهای اصلی، Benchmark/Gapfill/Sigmoid و IDEA-006. **WRITE:** فقط اسناد طراحی این پوشه، روی شاخه موقت. **RUN:** هیچ آموزش، Test یا تست اجرایی در این مرحله اجرا نشده؛ وضعیت همه تست‌ها NOT_RUN. **NEXT:** مدیر این گزارش و انتخاب objective/budget را ارزیابی کند؛ سپس پژوهشگر تست‌های سبک و پایلوت را تحت مجوز جداگانه پیاده کند. هیچ کد اصلی، raw قبلی، LaTeX یا شاخه main تغییر داده نشده است.

**درخواست رأی مدیر:** READY_FOR_PREFLIGHT_REVIEW؛ نه READY_FOR_PILOT و نه ACCEPTED.
