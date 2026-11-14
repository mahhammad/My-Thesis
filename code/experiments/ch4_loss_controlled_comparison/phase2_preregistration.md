# E-CH4-13 / Phase 2 — پیش‌ثبت مقایسه کنترل‌شده n=300

**وضعیت:** `PHASE2_DESIGN_ONLY / NO_FIT` — ۱۴ نوامبر ۲۰۲۶، RESEARCHER-06، Issue [#81](https://github.com/mahhammad/My-Thesis/issues/81). **ممیزی مستقل پایلوت:** Issue [#80](https://github.com/mahhammad/My-Thesis/issues/80) در هنگام پیش‌ثبت باز و بدون کامنت/رأی `ACCEPT_PILOT_SCOPED` بود. طراحی مجاز است؛ هر fit جدید ممنوع تا رأی مستقل و گیت تست.

## پرسش علمی و فرضیه پیشینی
آیا تغییر تابع زیان از `sigmoid_JS_margin_tau005` به `softplus_JS_tau005` در جفت‌هایی با موتور و داده یکسان، جهت `Δ Validation BA` و رابطه `R_KL/R_JS/R_sqrt_JS`, `I` و حاشیه را در مولدهای مستقل و دو policy خانواده حفظ می‌کند؟ فرض صفر توصیفی: نشانه و بزرگی دلتا میان مولد/خانواده ناپایدار است؛ هیچ inferential p-value/CI قوی یا «برتری عمومی» با سه مولد ادعا نمی‌شود.

## پروتکل قفل‌شده
- مولد: `code/data_generator/config.json` از main، فقط `dataset_adapter.dataset_train_val`؛ `data_seeds=[20261102,20261103,20261104]`، هر کدام Train=۳۰۰ و Validation=۹۰؛ هیچ تولید یا خواندن Test. پیکربندی اولیه و NumPy/OpenBLAS در manifest/fingerprint ثبت شود و هر تغییر/عدم تطابق STOP.
- `families=[shared,class]`؛ `objectives=[sigmoid_JS_margin_tau005,softplus_JS_tau005]`. برای class، مراکز Train بر اساس بازنمایی همان فرض کاندید؛ همه فرض‌ها برای Validation بدون اطلاع `y_val` محاسبه می‌شوند. `node/node_class` خارج دامنه.
- جفت هر `seed×family` با `reference_ids=0..299` ثابت (کل Train؛ هیچ زیرنمونه تصادفی)؛ `K=3,d=5`، ماسک و gamma از config همان بذر؛ `beta=.4,lambda=.8,H=12`، exact `float64`، logits صفر، Adam `lr=.12`، **۲۵ update برای هر fit**، `τ=.005` width، checkpoint endpoint=25. بدون Validation early stop، reselect یا optimize-on-Test.
- `sigmoid`: `mean_c mean_{i:y_i=c} σ(-(s_y-max_{r≠y}s_r)/τ)`؛ `softplus`: `mean_c mean_{i:y_i=c} τ log Σ_{r=0}^{K-1} exp((s_r-s_y)/τ)`. score `s_c=1−JSD(Q_{i,c},V_c)/ln 2`. کاهش متوازن کلاس دقیقاً یکسان؛ ولی سطح و norm/direction گرادیان دو هدف به‌طور ذاتی ناهم‌مقیاس است.
- grid کامل `3×2×2=12` خانه، سقف **۱۲ fit جدید** در مجموع؛ اجرا/تکرار شکست هم باید در دفتر بودجه شمارش شود، retry خودکار ممنوع. قبل از execute باید برپایه checksum موجودی تاریخی تصمیم قطعی reuse ثبت شود، نه بازچرخانی پنهان fit جدید. هر objective خارج دو مورد/seed خارج سه مورد/horizon یا budget متفاوت → خطا.
- هر گام ۰ تا ۲۵: value objective، gradient norm L2/Linf، norm به‌روزرسانی، cosine گرادیان بین دو loss در checkpointهای مشترک (در صورت ممکن از یک policy ثابت بدون optimizer اضافی)، train margin، elapsed؛ metadata dtype/library/BLAS/split/reference.
- دو snapshot خام `model.json` + `distributions.npz` برای هر fit، `trace.json`، `metrics_full.json` و `coverage.json`؛ یک `provenance.json` و manifest با SHA256 و no Test. `R_KL` support-aware/∞ دلیل‌دار، `R_JS`، `R_sqrt_JS=mean sqrt`، `I/T/BT/TNL/BTNL`، BA/Acc، per-class recall/confusion، ties، margin quantiles، همه NLL/surrogateهای موجود. جدول و شکل فقط از raw ساخته شود.
- دلتا جفتی `BA_sigmoid−BA_softplus` و اختلاف recall، flip count بین پیش‌بینی‌های Validation دو مدل همان seed/family، margin quantiles؛ برای هر family روی **سه مولد** mean و sample SD (ddof=1)، بدون تلقی تعداد Validation یا گام‌های optimizer به‌عنوان تکرار مستقل.
- Test مطلقاً ممنوع؛ مرحله پایان‌نامه و merge فقط پس از QA و رأی مدیر.

## ترتیب گیت و شرط توقف
1. `PHASE2_DESIGN_READY`: قرارداد، موجودی reuse، runner محافظت‌شده و fixture/tests بدون آموزش، ثبت SHA و checkpoint Issue #81.
2. در Issue #80 رأی مستقل `ACCEPT_PILOT_SCOPED` یا در حالت `ACCEPT_WITH_FIXES` رفع و قبول اصلاحات بحرانی **باید ثبت شده باشد**؛ نبود رأی یا REJECT → `NO_FIT`.
3. تمام preflightهای engine/origin/mask/gradient/tau/class multi-hypothesis/Train-only center/label leakage/KL/coverage/snapshot/fingerprint و budget/opt-in واقعی روی **همان SHA** PASS با log. `NOT_RUN` اجازه نیست.
4. فقط opt-in صریح همراه evidence و گیت فعال‌شده توسط مدیر، batch دقیقاً تا سقف ۱۲ fit. هر mismatch یا NaN/خرابی raw/سنجه → `STOP` و جلوگیری از retry. هر تلاش به‌محض ورود به optimizer در دفتر بودجه مصرف می‌شود.
5. `PHASE2_READY_FOR_INDEPENDENT_QA` با raw، checksum، گزارش علمی، provenance و جدول/شکل واقعی. تا پیش از QA هیچ نتیجه عام یا تغییر `tex/`.

## خطرات از پیش ثبت شده
- BLAS-dependent byte fingerprints در پایلوت: `OPENBLAS_CORETYPE=generic`, `OPENBLAS_NUM_THREADS=1`، NumPy=2.4.6، Torch CPU=2.15.0؛ **هر سه seed** باید با fingerprint و محیط قفل‌شده در raw ثبت شوند. اختلاف بایتی بی‌مجوز قابل اغماض نیست.
- برچسب Validation فقط در گزارشگر و محاسبه BA بعد از freeze سیاست/مرکز مجاز است؛ حضور در fit/ساخت prediction هرگز.
- خروجی تاریخی `n300×25` فقط در صورت تطبیق کامل داده، bytes/config، reference IDs، family، objective/τ، init/optimizer/steps/engine code و checkpoint قابل reuse است؛ در غیر این صورت `NOT_COMPARABLE`، نه حذف یا انتساب به phase2.
- استفاده از ۳ seed از یک generator، جای چند dataset واقعی یا آزمون تعمیم جامع را نمی‌گیرد. هر نتیجه باید تفسیر محدود داشته باشد.
