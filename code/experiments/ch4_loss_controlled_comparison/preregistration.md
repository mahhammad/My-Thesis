# E-CH4-13 — پیشنهاد پیش‌ثبت (تا تصویب مدیر اجرا نشود)
**Owner RESEARCHER-06 | Based on main@ac5892ad166ce300a2044e3303d4adaa1a21d7d7 | status PROPOSAL**

- پرسش: اختلاف paired Validation BA میان objectiveها تحت Train/Validation و start/optimizer/steps یکسان؛ آیا I↑ و R↓ به بهبود BA مربوط می‌شوند؟
- واحد مستقل: داده مصنوعی از seedهای 20261102/03/04؛ *نمونه‌های همان generator* و checkpointهایش واحد مستقل تازه نیستند.
- اولویت خانواده‌ها: shared و class با همه فرض‌های نامزد برون‌گراف. node/node_class بدون شروع معتبر خارج از دامنه.
- داده/موتور: K=3، d=5، n=300؛ Validation موجود هر seed؛ β=.4, λ=.8, H=12, float64 و exact، gamma و reference IDs frozen از config/manifest.
- فقط پس از تصویب: سه objective اول sigmoid JSD τ=.005 / Softplus JS τ=.005 / KL normalized historical NLL؛ چهارمی NLL_KL_simple فقط با رأی مدیر. baseline uniform و raw تاریخی reuse فقط پس از fingerprint + checksum.
- آموزش: Adam lr=.12، ۲۵ گام، paired zero-init؛ هیچ انتخاب checkpoint/τ با Test. اگر random init در دور بعد اضافه شود، seed_init مستقل، یکسان در pair و گزارش صریح.
- گسترش KL margin / sqrtJS margin / maxI / combined I-R-margin، τ=.002/.020، چندشروعی/گرادیان تک‌نمونه **در این طرح خودکار مجاز نیست**.
- ملاک اصلی: paired Δ Validation BA در سطح seed generator. ثانوی: recall/class, confusion, ties, true margin quantiles, R_KL exact, R_JS, R_sqrt_JS, I/T/BT/TNL/BTNL, NLL/Softplus/sigmoid, time and gradient norm.
- aggregate: میانگین و SD توصیفی اختلاف‌های paired برای هر خانواده؛ سه seed برای CI قوی کافی نیست. عدم اختلاف و شکست نیز نگهداری شوند.
- کنترل خطر: validation label perturbation استقلال logits/pred؛ no Test; stop on nonfinite train gradient/loss, bad simplex, unsupported KL ambiguity, checksum/config mismatch, mask/origin mismatch, undefined query start.
- مرحله اول پیشنهادی *بعد از گیت*: فقط ۲ fit (shared × sigmoid/softplus، seed=20261102، n=50، ۱۰ گام، Adam .12، init صفر). پیش از هر fit تمام fixtureهای lightweight باید PASS واقعی داشته باشند. بودجه پیشنهادی **مجوز فعلی اجرا نیست**.
- نتایج و گزارش: immutable model.json/distributions.npz + sha256, full metrics, coverage reason codes, trace/manifest, independent tests, tables and figures from same data source.
- state: ALL TESTS NOT_RUN; NO TRAINING; NO TEST; READY_FOR_PREFLIGHT_REVIEW requested.
