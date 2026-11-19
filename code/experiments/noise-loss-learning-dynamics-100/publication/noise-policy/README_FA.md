# مواجههٔ سیاست با نویز — شاخص‌های اختصاصی §۳.۸

## تعریف و دامنه

این تحلیل از سیاستِ ثبت‌شده در هر وضعیت آموزش t=0,...,100 در [Run #44419173443](https://github.com/mahhammad/My-Thesis/actions/runs/44419173443) محاسبه می‌شود. مبنای آن **فرایند کمکی بدون ماسک** با سطر شروع p0 و ماتریس انتقال P است؛ این سنجه‌ها برآورد مستقیم مواجهه در گشت واقعیِ ماسک‌دار نیستند.

مجموعهٔ غیرنویزی A_P شامل سه ویژگی مرجع است. تعداد نویز فقط در سطوح |A_N|=1,...,6 در شکل‌های مقایسه‌ای وارد می‌شود. برای هر سطح شش زیرمجموعهٔ وابسته داریم و در خانوادهٔ class ابتدا شاخص برای هر فرض کلاس محاسبه می‌شود و سپس میانگین برابر‌وزن کلاس‌ها به‌عنوان خلاصهٔ توصیفی ارائه می‌شود.

## شاخص‌های گزارش‌شده با نماد رسمی فصل سوم

- q_t = p0 P^(t-1) 1_(A_N)، برای نوبت‌های ۱،۲،۳
- p_hit,N^(h): احتمال حداقل یک برخورد با نویز تا افق h (به‌ویژه h=2 و h=3)
- nbar_N^(h): تعداد مورد انتظار انتخاب‌های نویزی تا افق h
- s_N^(h) = nbar_N^(h)/h: سهم مورد انتظار نویز به‌ازای هر نوبت
- s_P^(h) = 1 - s_N^(h): سهم غیرنویز در همان فرایند
- NFI_N^(3) = p_hit,N^(3) / p_hit,N^(3),U
- NSR_N^(h) = s_N^(h) / (|A_N|/d)، برای h=2,3

مرجع سیاست یکنواخت هم‌بُعد برابر s_N^(h),U = |A_N|/d و p_hit,N^(h),U = 1-(|A_P|/d)^h است. مقدار ۱ برای NFI یا NSR به معنی برابری با مرجع یکنواخت است، نه احتمال صددرصد. نسبت‌ها برای سطح نویز صفر تعریف نمی‌شوند و بنابراین نمودارهای مقایسه‌ای فقط سطوح ۱ تا ۶ را شامل می‌شوند.

## شکل‌های مورد نیاز

برای هر خانوادهٔ shared و class و هر زیان Sigmoid-JS و Softplus-JS، مسیر کامل شاخص‌های q1، q2، q3، p_hit,N^(2)، p_hit,N^(3)، nbar_N^(3)، s_N^(2)، s_N^(3)، s_P^(3)، NFI_N^(3)، NSR_N^(2) و NSR_N^(3) به‌صورت جداگانه رسم می‌شود. در هر نمودار، پنل‌های شش سطح نویز و منحنی‌های منفرد کم‌رنگ به‌همراه میانگین پررنگ قرار می‌گیرند.

برای مقایسهٔ سهم ویژگی‌های غیرنویزی و نویزی باید مقادیر s_N و s_P هر دو نمایش داده شوند؛ مقدار سهم غیرنویزی با کم‌کردن سهم نویز از یک به دست می‌آید، نه با محاسبهٔ مجدد در گراف ماسک‌دار.

نام‌های فایل خروجی بر اساس خانواده، تابع زیان و شاخص (برای مثال shared__sigmoid__s3.svg) هستند. فایل تجمعی noise_policy_101_steps.csv برای ممیزی عددی تمام مسیرها نگه داشته می‌شود.

## ملاحظات پژوهشی

وجود احتمال مثبت نویز به علت Softmax الزاماً به معنی مفیدبودن آن نیست. تغییر شاخص‌ها در طول یادگیری را نباید به‌تنهایی علت تغییر BA دانست. معیارهای این بخش، انتخاب ویژگی در سیاست بدون ماسک را توصیف می‌کنند. تعریف کامل و مرجع یکنواخت در §۳.۸ و معادلات برچسب‌دار eq:ch3-noise-qt، eq:ch3-noise-general-horizon، eq:ch3-noise-normalized-expectation و eq:ch3-noise-normalized-uniform-ratio آمده‌اند.

**وضعیت:** این فایل قرارداد گزارش علمی و فهرست خروجی‌هاست. بارگذاری تصاویر مرحله‌ای مستقل است و نباید پیش از انتشار آن‌ها تکمیل‌شده تلقی شود.

## نمایهٔ شکل‌های ثبت‌شده در همین پوشه

[دادهٔ کامل ۱۰۱ گام (CSV)](noise_policy_101_steps.csv)

### سیاست مشترک ـ زیان سیگمویدی

- احتمال انتخاب نویز در نوبت اول: [شکل برداری](shared__sigmoid__q1.svg) · [تصویر PNG](shared__sigmoid__q1.png)
- احتمال انتخاب نویز در نوبت دوم: [شکل برداری](shared__sigmoid__q2.svg) · [تصویر PNG](shared__sigmoid__q2.png)
- احتمال انتخاب نویز در نوبت سوم: [شکل برداری](shared__sigmoid__q3.svg) · [تصویر PNG](shared__sigmoid__q3.png)
- احتمال برخورد حداقل یک‌باره تا افق ۲: [شکل برداری](shared__sigmoid__hit2.svg) · [تصویر PNG](shared__sigmoid__hit2.png)
- احتمال برخورد حداقل یک‌باره تا افق ۳: [شکل برداری](shared__sigmoid__hit3.svg) · [تصویر PNG](shared__sigmoid__hit3.png)
- تعداد مورد انتظار انتخاب‌های نویزی: [شکل برداری](shared__sigmoid__nbar3.svg) · [تصویر PNG](shared__sigmoid__nbar3.png)
- سهم نویز در دو نوبت: [شکل برداری](shared__sigmoid__s2.svg) · [تصویر PNG](shared__sigmoid__s2.png)
- سهم نویز در سه نوبت: [شکل برداری](shared__sigmoid__s3.svg) · [تصویر PNG](shared__sigmoid__s3.png)
- سهم غیرنویز در سه نوبت: [شکل برداری](shared__sigmoid__sp3.svg) · [تصویر PNG](shared__sigmoid__sp3.png)
- نسبت احتمال برخورد به یکنواخت: [شکل برداری](shared__sigmoid__NFI3.svg) · [تصویر PNG](shared__sigmoid__NFI3.png)
- نسبت سهم نویز به یکنواخت در افق ۲: [شکل برداری](shared__sigmoid__NSR2.svg) · [تصویر PNG](shared__sigmoid__NSR2.png)
- نسبت سهم نویز به یکنواخت در افق ۳: [شکل برداری](shared__sigmoid__NSR3.svg) · [تصویر PNG](shared__sigmoid__NSR3.png)

### سیاست مشترک ـ زیان سافت‌پلاس

- احتمال انتخاب نویز در نوبت اول: [شکل برداری](shared__softplus__q1.svg) · [تصویر PNG](shared__softplus__q1.png)
- احتمال انتخاب نویز در نوبت دوم: [شکل برداری](shared__softplus__q2.svg) · [تصویر PNG](shared__softplus__q2.png)
- احتمال انتخاب نویز در نوبت سوم: [شکل برداری](shared__softplus__q3.svg) · [تصویر PNG](shared__softplus__q3.png)
- احتمال برخورد حداقل یک‌باره تا افق ۲: [شکل برداری](shared__softplus__hit2.svg) · [تصویر PNG](shared__softplus__hit2.png)
- احتمال برخورد حداقل یک‌باره تا افق ۳: [شکل برداری](shared__softplus__hit3.svg) · [تصویر PNG](shared__softplus__hit3.png)
- تعداد مورد انتظار انتخاب‌های نویزی: [شکل برداری](shared__softplus__nbar3.svg) · [تصویر PNG](shared__softplus__nbar3.png)
- سهم نویز در دو نوبت: [شکل برداری](shared__softplus__s2.svg) · [تصویر PNG](shared__softplus__s2.png)
- سهم نویز در سه نوبت: [شکل برداری](shared__softplus__s3.svg) · [تصویر PNG](shared__softplus__s3.png)
- سهم غیرنویز در سه نوبت: [شکل برداری](shared__softplus__sp3.svg) · [تصویر PNG](shared__softplus__sp3.png)
- نسبت احتمال برخورد به یکنواخت: [شکل برداری](shared__softplus__NFI3.svg) · [تصویر PNG](shared__softplus__NFI3.png)
- نسبت سهم نویز به یکنواخت در افق ۲: [شکل برداری](shared__softplus__NSR2.svg) · [تصویر PNG](shared__softplus__NSR2.png)
- نسبت سهم نویز به یکنواخت در افق ۳: [شکل برداری](shared__softplus__NSR3.svg) · [تصویر PNG](shared__softplus__NSR3.png)

### سیاست وابسته به کلاس ـ زیان سیگمویدی

- احتمال انتخاب نویز در نوبت اول: [شکل برداری](class__sigmoid__q1.svg) · [تصویر PNG](class__sigmoid__q1.png)
- احتمال انتخاب نویز در نوبت دوم: [شکل برداری](class__sigmoid__q2.svg) · [تصویر PNG](class__sigmoid__q2.png)
- احتمال انتخاب نویز در نوبت سوم: [شکل برداری](class__sigmoid__q3.svg) · [تصویر PNG](class__sigmoid__q3.png)
- احتمال برخورد حداقل یک‌باره تا افق ۲: [شکل برداری](class__sigmoid__hit2.svg) · [تصویر PNG](class__sigmoid__hit2.png)
- احتمال برخورد حداقل یک‌باره تا افق ۳: [شکل برداری](class__sigmoid__hit3.svg) · [تصویر PNG](class__sigmoid__hit3.png)
- تعداد مورد انتظار انتخاب‌های نویزی: [شکل برداری](class__sigmoid__nbar3.svg) · [تصویر PNG](class__sigmoid__nbar3.png)
- سهم نویز در دو نوبت: [شکل برداری](class__sigmoid__s2.svg) · [تصویر PNG](class__sigmoid__s2.png)
- سهم نویز در سه نوبت: [شکل برداری](class__sigmoid__s3.svg) · [تصویر PNG](class__sigmoid__s3.png)
- سهم غیرنویز در سه نوبت: [شکل برداری](class__sigmoid__sp3.svg) · [تصویر PNG](class__sigmoid__sp3.png)
- نسبت احتمال برخورد به یکنواخت: [شکل برداری](class__sigmoid__NFI3.svg) · [تصویر PNG](class__sigmoid__NFI3.png)
- نسبت سهم نویز به یکنواخت در افق ۲: [شکل برداری](class__sigmoid__NSR2.svg) · [تصویر PNG](class__sigmoid__NSR2.png)
- نسبت سهم نویز به یکنواخت در افق ۳: [شکل برداری](class__sigmoid__NSR3.svg) · [تصویر PNG](class__sigmoid__NSR3.png)

### سیاست وابسته به کلاس ـ زیان سافت‌پلاس

- احتمال انتخاب نویز در نوبت اول: [شکل برداری](class__softplus__q1.svg) · [تصویر PNG](class__softplus__q1.png)
- احتمال انتخاب نویز در نوبت دوم: [شکل برداری](class__softplus__q2.svg) · [تصویر PNG](class__softplus__q2.png)
- احتمال انتخاب نویز در نوبت سوم: [شکل برداری](class__softplus__q3.svg) · [تصویر PNG](class__softplus__q3.png)
- احتمال برخورد حداقل یک‌باره تا افق ۲: [شکل برداری](class__softplus__hit2.svg) · [تصویر PNG](class__softplus__hit2.png)
- احتمال برخورد حداقل یک‌باره تا افق ۳: [شکل برداری](class__softplus__hit3.svg) · [تصویر PNG](class__softplus__hit3.png)
- تعداد مورد انتظار انتخاب‌های نویزی: [شکل برداری](class__softplus__nbar3.svg) · [تصویر PNG](class__softplus__nbar3.png)
- سهم نویز در دو نوبت: [شکل برداری](class__softplus__s2.svg) · [تصویر PNG](class__softplus__s2.png)
- سهم نویز در سه نوبت: [شکل برداری](class__softplus__s3.svg) · [تصویر PNG](class__softplus__s3.png)
- سهم غیرنویز در سه نوبت: [شکل برداری](class__softplus__sp3.svg) · [تصویر PNG](class__softplus__sp3.png)
- نسبت احتمال برخورد به یکنواخت: [شکل برداری](class__softplus__NFI3.svg) · [تصویر PNG](class__softplus__NFI3.png)
- نسبت سهم نویز به یکنواخت در افق ۲: [شکل برداری](class__softplus__NSR2.svg) · [تصویر PNG](class__softplus__NSR2.png)
- نسبت سهم نویز به یکنواخت در افق ۳: [شکل برداری](class__softplus__NSR3.svg) · [تصویر PNG](class__softplus__NSR3.png)

