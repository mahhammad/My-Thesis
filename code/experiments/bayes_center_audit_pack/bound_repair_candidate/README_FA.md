# CH4-D10D11-BOUND-REPAIR-01 — نامزد بازسازی کران‌های بلمن

**مالک:** `RESEARCHER-02` · **مرجع آغاز شاخه:** `chapter4@de02fdda91e23db177de4429b3e1a9693c046d83`. **وضعیت:** نامزد اصلاح، نیازمند ممیزی مستقل؛ نه اجرای گراف ۳۰۰تایی و نه مجوز نوشتن §۴.۱۱.

**علت:** [ممیزی مستقل REVISION_REQUIRED](https://github.com/mahhammad/My-Thesis/issues/49#issuecomment-6794025754) نشان داد که هستهٔ بازگشت‌مجاز قدیمیِ `origin_return_relaxed_bounds` یک کران خارجی معتبر برای هستهٔ مبدأحذف نیست. [تصمیم مدیر](https://github.com/mahhammad/My-Thesis/issues/49#issuecomment-6794350204) تا اصلاح و QA مستقل، ادعاهای قطعی D10/D11 را در `SCIENTIFIC_HOLD` نگه داشته است.

## فایل‌ها

- [`PROOF_AND_ASSUMPTIONS_FA.md`](PROOF_AND_ASSUMPTIONS_FA.md): اثبات مستقل شمول و فرض‌های دقیق فصل ۳، شناسایی هستهٔ صحیح و مثال نقض.
- [`exact_origin_excluded_bound.py`](exact_origin_excluded_bound.py): پیاده‌سازی **جدید** کم‌حافظه `exact_origin_excluded_center_bounds(x,labels,gamma,alpha)` با O(d n²) حافظه؛ هستهٔ `q_b^(s)` **با حذف مبدأ و بازنرمال‌سازی**، نه هستهٔ بازگشت‌مجاز B؛ `kernel_version=exact_origin_excluded_low_memory_candidate_v1`.
- [`test_exact_origin_excluded_bound.py`](test_exact_origin_excluded_bound.py): شش آزمون محدود به `n≤6`، با import تابع **اصلی آرشیوی** `bounds_for_all_origins` در checkout کامل به عنوان oracle؛ کنترل fixed Z با masked softmax و مقصدهای نامعتبر.
- [`tiny_validation.py`](tiny_validation.py): اسکریپت گزارش‌ساز اجرای **فقط toy**، seed `20261111`؛ ارزیابی خطا/زمان/RSS و اجرای شش unit-test. اجرای داده‌های پایان‌نامه یا grid را ممکن نمی‌کند.
- [`tiny_test_results.json`](tiny_test_results.json): نتیجه عددی fixtureهای `n=3,4,6`، حداکثر اختلاف با oracle، `0.36` در برابر `71/375`، seed و زمان/RSS.
- [`CLAIM_STATUS_AND_D11_RECHECK_FA.md`](CLAIM_STATUS_AND_D11_RECHECK_FA.md): نگاشت C01–C10 به `CERTIFIABLE_AFTER_QA/HISTORICAL_ONLY/ON_HOLD`؛ برنامه آزمون دوبارهٔ اهداف D11 با کران **تازه**.
- [`COST_AND_QA_PLAN_FA.md`](COST_AND_QA_PLAN_FA.md): حافظه/زمان نظری، دو برنامهٔ جدا full300×37 و subset72×37، گیت‌های اجرایی پس از مجوز.

## بازتولید آزمون‌های سبک (فقط از clone کامل، Python/NumPy و Linux)

```bash
python -m unittest discover -s code/experiments/bayes_center_audit_pack/bound_repair_candidate -p 'test_*.py' -v
python code/experiments/bayes_center_audit_pack/bound_repair_candidate/tiny_validation.py \
  --out /tmp/ch4_bound_repair_tiny_test_results.json
```

اسکریپت دوم فقط سه fixture تعریف‌شدهٔ `n=3`, `n=4`, `n=6` و شش آزمون محلی را اجرا می‌کند؛ منبع oracle `code/experiments/archive/bayes_center_inverse/feasibility_bound.py` **بدون تغییر** خوانده می‌شود. در محیط اجرای تهیه‌کننده، مخزن کامل با copy محلیِ **متن تابع آرشیوی** برای آزمون ساخته شده بود؛ در اجرای مستقل روی checkout واقعی، تست خود فایل قفل‌شدهٔ مخزن را import خواهد کرد. blob آرشیوی: `79251967ed0d799185d502c8da44548eddd6e3ed`. این تفاوت شیوه بارگذاری به‌عنوان مورد کنترل QA گزارش می‌شود، نه ادعای QA مستقل قبلی.

### مشاهدات fixtureهای کوچک

- **شش تست PASS**؛ کمینه/بیشینه per-origin و class-mean در هر fixture با DP مرجع تا حداکثر `5.551115123125783e-17` اختلاف دارند؛ کنترل fixed-policy شاهدی بیرون از بازه نیافته است.
- در مثال ممیز `n=3,d=1,H=2`, کران اصلاح‌شده و مقدار گشت اصلی در مختصه موردنظر برابر `0.36` (خطای گردکردن `5.55e-17`)؛ کران نامعتبر قدیمی `71/375≈0.189333` است.
- **اعداد این اجرا:** زمان داخلی آزمون/تطبیق تقریباً `0.0456s` و RSS فرایند Python تقریباً `98188 KiB = 0.0936 GiB`. اندازه‌گیری `/usr/bin/time` زمان کل با importها را `0.79s` و RSS نهایی `98316 KiB` نشان داده است؛ کل هزینهٔ عددی زیر سقف مأموریت (`120s`, `1GiB`) بود. این اعداد هرگز زمان/حافظه full300 را پیش‌بینی نمی‌کنند.

## خط قرمزها

**هیچ** فایل `feasibility_bound.py`، خروجی JSON/CSV قدیمی، historical ZIP، فصل ۳/۴، `main` یا `chapter4` تغییر داده نشده. **هیچ** اجرای n300، شبکه ۳۷ تنظیم، solver، optimizer، MC، آموزش، Test/Validation query، PR یا merge صورت نگرفته است. در **فیلدهای متادیتای خروجی تابع جدید** برچسب `full300_return_allowed` به‌کار نمی‌رود؛ در صورت اجرای جداگانه، برچسب دقیق `exact_origin_excluded...` باید حفظ شود.

**وضعیت علمی فعلی:** نمونه نقض فقط *گواه تاریخی* را باطل کرده است؛ نامزد جدید برهان و تست toy دارد ولی **تأیید مستقلی برای اعداد D10/D11 وجود ندارد**. متن بخش ۴.۱۱ تا گیت QA و دستور مدیریت در حالت `SCIENTIFIC_HOLD` می‌ماند.
