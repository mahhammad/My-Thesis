# بازیابی آزمایش‌های تاریخی — وضعیت آرشیو

> **این فایل فقط فهرست و اثرانگشت بسته‌های اصلی است، نه جایگزین کد و خروجی خام.**
> نسخه اصلی فصل چهارم پیش از بازآرایی: `backup/chapter4-before-benchmark-v1-20261108` @ `72ed366dd84a0dd56acd55ddf48c9130eedbf655`.

در ۸ نوامبر ۲۰۲۶ هر نه بسته زیر در محیط آزمایش بازیابی شدند و آزمون سلامت ZIP گذراندند. **تمام نه بستهٔ ZIP اصلی در `historical_archive/archives/` ثبت شده‌اند.** checksumهای SHA-256 زیر برای تطبیق بایت‌به‌بایت استفاده می‌شوند. این بسته‌ها بازتولیدپذیری تاریخی را حفظ می‌کنند، اما خودِ گزارش‌های پیش از استاندارد v1 محسوب می‌شوند. بسته‌های اصلی، اسکریپت‌ها و JSONها را با همان ساختار داخل ZIP حفظ می‌کنند. استخراج انتخابی فایل‌های مرجع در مسیر فعال در مرحله بعدی انجام می‌شود.

| نام بسته | بایت | SHA-256 | وضعیت |
|---|---:|---|---|
| `multistep_policy_target_match_full.zip` | 32361 | `9ad828303636d5fd41c7792c427ef70c49ccd6e8c567f5078e8cd742514ab590` | archived in `archives/` |
| `center_inverse_bootstrap_experiment.zip` | 49787 | `55cd1602a92b939c42a61247b263d3326651175df74920d10d398d984e141303` | archived in `archives/` |
| `full300_maxdiag_bootstrap_experiment.zip` | 57083 | `6881452b41f80957c035d29f68e4b723c0d9f2839776a2585b322eaf4c94f26b` | archived in `archives/` |
| `margin_sigmoid_only_experiment.zip` | 81455 | `45745836de9c351a86a732566d3584daa63a1305574c7321db5d6fc3eaa27635` | archived in `archives/` |
| `train_size_margin_only_complete.zip` | 63963 | `d952560a2b22607e0b08e838723acf7bd2046721c90cf6c2e7bd4126c4e7c256` | archived in `archives/` |
| `inverse_margin_center_class_conditional.zip` | 51248 | `331b2cc040fefcfd01a4d207dd9b299fe4bbdebb3874f24fa48953110c065cdb` | archived in `archives/` |
| `policy_comparison_complete.zip` | 35416 | `c127d2cff80c7c7fbf2858054c80c79324412c8bc4fccdcafe0461af421e73d1` | archived in `archives/` |
| `fixed_center_compaction_complete.zip` | 69599 | `60075b025b74abb520e16e5f08810a3860d9d2b4e23cdeebf7c727f7138e22f7` | archived in `archives/` |
| `margin_fixed_centers_complete.zip` | 67537 | `dc689ea9c517469c7fc56377d9a50c2ecd3cdf22d26682251323e6e55bc03d0e` | archived in `archives/` |

گزارش‌های موجودِ GitHub مربوط به بسته‌های اخیر در `inverse_margin_center/`, `policy_architecture_comparison/`, `fixed_center_compaction/` و `fixed_center_margin/` هستند و جای خروجی‌های کامل را نمی‌گیرند. همه بازاجراهای جدید باید زیر `benchmark_v1` با `run_manifest.json` و سطح `benchmark-v1` ذخیره شوند؛ گزارش‌های قدیمی فقط `historical` محسوب می‌شوند. هیچ داده تاریخی پاک یا بی‌صدا اصلاح نمی‌شود.


## انتقال بایت‌به‌بایت به GitHub (۸ نوامبر ۲۰۲۶)

همه بسته‌ها از فایل خام ZIP به base64 برای بارگذاری blob منتقل شده‌اند؛ **فایل ذخیره‌شده در GitHub خود ZIP باینری است** و نه فایل متنی base64. از این پس منابع تاریخی برای بررسی و بازتولید مستقیماً در همان برنچ موجودند. نگه‌داری آنها جایگزین اجرای تازه Benchmark v1 نیست.
