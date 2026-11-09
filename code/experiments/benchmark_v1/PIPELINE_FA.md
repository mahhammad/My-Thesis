# قرارداد پایپ‌لاین خروجی و گزارش IMC — نسخه ۱

این ابزار با الگوی [Data Generator](../../data_generator/README.md) ساخته شده: **ورودی ثابت، خروجی ثابت، و جدایی محاسبه مدل از گزارش**.

**دستاورد:** اگر بعداً معیار یا قالب گزارش را تغییر دادیم، از بازنمایی‌های ذخیره‌شده گزارش جدید تولید می‌کنیم؛ نیازی به آموزش مجدد یا محاسبه دوباره گشت‌ها نیست.

## معماری

```text
داده مرجع + سیاست + پارامترهای گشت
                  |
        policy_interface.py
                  |
       snapshot_pipeline.py
                  |
           دو فایل خام
         /             \
   model.json      distributions.npz
         \             /
           report_snapshot.py
                  |
            report.json
             report.md

کد هر آزمایش -> گزارش تخصصی همان آزمایش
                experiment_report.json
```

**گزارش تخصصی متعلق به کد همان آزمایش است، نه گزارشگر عمومی.** مثلاً کد بازیابی مراکز، جداگانه خطای مرکز هدف را اندازه می‌گیرد؛ گزارش عمومی فقط معیارهای مشترک را حساب می‌کند.

## ورودی تابع عمومی سیاست

[`policy_interface.py`](policy_interface.py) یک رابط مشترک برای سیاست‌های مختلف دارد:

```python
result = evaluate_policy(
    dataset={"ref_x": X_train, "ref_y": y_train, "query_x": X_validation},
    policy_family="shared",
    policy_parameters=Z,
    gamma=gamma, beta=0.4, lam=0.8, horizon=12
)
```

این تابع آموزش، محاسبه زیان یا گزارش‌نویسی نمی‌کند؛ فقط توزیع‌های برچسب همسایگی و مراکز را تولید می‌کند و **برای نمونه Validation برچسب واقعی نمی‌گیرد**.

| خانواده | پارامتر سیاست | استنتاج خارج از Train |
|---|---|---|
| `shared` | `(d+1,d)` | تعریف‌شده |
| `class` | `(K,d+1,d)` | همه فرض‌های کلاس جدا محاسبه می‌شوند |
| `node` | `(n,d+1,d)` | بدون سیاست شروع مستقل برای query تعریف‌نشده |
| `node_class` | `(K,n,d+1,d)` | بدون سیاست شروع مستقل برای query تعریف‌نشده |
| `class_tied_node` | `start_by_class: (K,d)` و `intermediate_by_node_class: (K,n,d,d)` | با شروع مشترک کلاسی تعریف‌شده؛ با خانواده آزاد اشتباه نشود |

برای خانواده‌های آزاد می‌توان `query_start_logits` را به همراه `query_start_source` صریحاً وارد کرد؛ هرگز نباید از برچسب واقعی query برای انتخاب فرض کلاس استفاده شود.

## قرارداد دو فایل خام

| فایل | محتوا | سیاست نگهداری |
|---|---|---|
| `model.json` | logits کامل، نام/نوع سیاست، مشخصات داده و بذرها، شناسه مرجع، kernel gamma، beta/lambda/H، هدف و tau، optimizer و بودجه موجود، هش فایل‌ها و منشأ checkpoint | بعد از ایجاد **تغییرناپذیر** |
| `distributions.npz` | `train_by_hypothesis`، `validation_by_hypothesis`، `centers`، `reference_ids` | تغییرناپذیر و دارای SHA-256 |

قرارداد آرایه‌ها:

- `train_by_hypothesis` با شکل `(n,K,K)`: عنصر `[i,c,r]` احتمال برچسب مقصد `r` برای نمونه `i` تحت **فرض نامزد `c`** است.
- `validation_by_hypothesis` با شکل `(m,K,K)`، اگر شروع query تعریف شده باشد؛ وگرنه `(0,K,K)` و توضیح `not_defined_without_query_start`. صفر ساختگی برای Validation ممنوع است.
- `centers` با شکل `(K,K)`: مرکز `c` برابر **میانگین حسابی Train** از `train_by_hypothesis[i,c,:]` برای تمام نمونه‌های کلاس `c` است.
- `reference_ids` اندیس‌ها را نسبت به Train مولد و به همان ترتیب دقیق حفظ می‌کند.

**چرا همه K فرض را ذخیره می‌کنیم؟** تا گزارشگر بتواند بعدها بدون اجرای گشت، امتیاز هر رقیب، حاشیه، خطا و نمودارهای جدید را به دست آورد. در سیاست مشترک، برش‌های فرض کلاس یکسان‌اند؛ در سیاست شرطی می‌توانند متفاوت باشند.

برچسب واقعی در پیش‌بینی وارد نمی‌شود. برای گزارش، برچسب‌های Train و Validation جداگانه از مولد و بذر اصلی بازیابی و با fingerprint ثبت‌شده تطبیق داده می‌شوند. این کار **آموزش یا گشت جدید نیست**، اما برای گزارش مجدد باید نسخه مولد و config مرجع موجود باشند.

## گزارش عمومی و تخصصی

| خروجی مشتق‌شده | مسئول | چه چیزی ثبت می‌کند |
|---|---|---|
| `report.json` | [`report_snapshot.py`](report_snapshot.py) | BA، Accuracy، I، سه R، حاشیه، ماتریس آشفتگی، NLLهای تشخیصی، خروجی‌های هر نمونه |
| `report.md` | همان | خلاصه خوانای Markdown |
| `experiment_report.json` | **کد همان آزمایش** | مثلاً JSD مراکز هدف، بیشینه residual، فاصله فروبنیوس؛ یا معیار اختصاصی آزمایش دیگر |

برای اضافه‌کردن معیار عمومی، فقط `report_snapshot.py` تغییر می‌کند و گزارش‌های مشتق‌شده بازنویسی می‌شوند. **دو فایل خام باقی می‌مانند.**

برای آزمون **حساسیت پهنای حاشیه** نیز می‌توان فقط گزارش را با `--tau 0.020` بازساخت؛ مقدار اصلی در `model.json` ثابت می‌ماند و گزارش جدید `metric_tau_source=report_override` ثبت می‌کند.

نمونه گزارش اختصاصی بازیابی مراکز: [`experiment_center_target.py`](experiment_center_target.py). این فایل جزئی از منطق همان آزمایش است؛ گزارشگر مشترک آن را صدا نمی‌زند.

## اجرای برنامه — بدون آموزش

از ریشه مخزن:

```bash
# فقط پیش‌نمایش؛ هیچ مدل اجرا نمی‌شود
python code/experiments/benchmark_v1/export_snapshot.py \
  --uniform --run-dir /tmp/imc_uniform

# تولید دو فایل خام از سیاست یکنواخت؛ بدون حتی یک گام آموزش
python code/experiments/benchmark_v1/export_snapshot.py \
  --uniform --execute --run-dir /tmp/imc_uniform

# محاسبه/بازمحاسبه گزارش از فایل‌های خام؛ بدون اجرای گشت
python code/experiments/benchmark_v1/report_snapshot.py \
  --run-dir /tmp/imc_uniform
```

پشتیبانی از checkpoint قبلی:

```bash
python code/experiments/benchmark_v1/export_snapshot.py \
  --checkpoint-json /path/to/checkpoint.json \
  --run-dir /tmp/imc_checkpoint --execute
```

یا یک فایلِ مانیفست‌دار در ZIP آرشیوی:

```bash
python code/experiments/benchmark_v1/export_snapshot.py \
  --archive code/experiments/benchmark_v1/results/core_50_16runs.zip \
  --member ds20261102_n300_shared_margin_JSD_tau0.005_s50.json \
  --run-dir /tmp/imc_old --execute
```

برای گزارش تخصصی بازیابی مرکز، خود کد آن آزمایش پس از `save_snapshot(...)` باید `save_center_target_report(run_dir)` را فراخوانی کند. برای snapshot موجود دارای مراکز هدف هم قابل استفاده است:

```bash
python code/experiments/benchmark_v1/experiment_center_target.py \
  --run-dir /tmp/imc_center_run
```

## صحت‌سنجی و محدودیت

- هیچ سیاستی حق استفاده از **برچسب واقعی Validation** برای انتخاب فرض کلاس را ندارد.
- مراکز باید با میانگین واقعی Train سازگار باشند؛ Validation آن‌ها را دوباره تخمین نمی‌زند.
- نمونه‌های آزاد `node/node_class` بدون قانون شروع query باید Validation نامعین داشته باشند.
- ماتریس احتمال‌ها باید متناهی و سطرهایش با جمع یک باشد؛ فایل خام با checksum کنترل می‌شود.
- گزارش R_KL نامتناهی را با مقدار JSON نامعتبر ذخیره نمی‌کند؛ به‌صورت `null` و شمار رخداد گزارش می‌کند.
- تعریف NLL نرمال‌شده تاریخی از هموارسازی eta/epsilon فصل سوم متفاوت است و جدا نام‌گذاری می‌شود.
- هر اجرا پوشه مستقل دارد؛ ابزار snapshot روی فایل خام موجود نمی‌نویسد، اما **گزارش مشتق‌شده قابل بازنویسی است**.
- قبل از تبدیل به استاندارد اجباری تمام آزمایش‌ها، تست سبک زیر را اجرا و نتیجه را ثبت کنید:

```bash
python -m unittest -v code/experiments/benchmark_v1/test_snapshot_pipeline.py
```

**وضعیت:** پیاده‌سازی و تست‌ها روی برنچ `chapter4` ثبت شده‌اند؛ **تست‌ها در این نشست اجرا نشده‌اند** و هیچ آزمایش سنگین یا ارزیابی Test تاریخی انجام نشده است.
