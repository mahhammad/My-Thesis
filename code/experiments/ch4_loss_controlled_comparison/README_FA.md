# E-CH4-13 — مقایسه کنترل‌شده تابع زیان IMC

**مالک:** RESEARCHER-06 · **مدیر:** RESEARCH-MANAGER-04 · [Issue #78](https://github.com/mahhammad/My-Thesis/issues/78) · [Draft PR #79](https://github.com/mahhammad/My-Thesis/pull/79)

## وضعیت جاری: PILOT_READY_FOR_QA / UNVERIFIED SCIENTIFIC CLAIMS
طرح پیش‌پرواز [پذیرفته‌شده با شروط](https://github.com/mahhammad/My-Thesis/issues/78#issuecomment-6880231186) و **دقیقاً دو آموزش مجاز، هرکدام ۱۰ update** روی seed=20261102 و shared انجام شد. گزارشگر مرحله نهایی در اجرای اولیه خطای پوشش متریک داشت؛ فایل‌های هر دو run به‌صورت artifact محفوظ ماندند و بدون اجرای optimizer مجدد [بازیابی و راستی‌آزمایی شدند](https://github.com/mahhammad/My-Thesis/actions/runs/43615046534).

**مطالعه کامل چندبذری، full300، Test و merge هنوز مجاز نیست.** نتیجه مقدماتی فقط مربوط به یک مولد و بودجه کم است؛ QA مستقل علمی انجام نشده است.

## مسیر مطالعه و تحویل
- [RESEARCHER-06_READINESS_REPORT.md](RESEARCHER-06_READINESS_REPORT.md): گزارش پیش‌پرواز اولیه (checkpoint تاریخی قبل از پایلوت)
- [preregistration.md](preregistration.md): طرح و فرضیه کنترل‌شده
- [REPORT_FA.md](REPORT_FA.md): گزارش کامل پایلوت، نتایج، شواهد منفی و حدود استنباط
- [inventory_existing.csv](inventory_existing.csv): موجودی مقایسه‌های سابق
- [PILOT_EXECUTION_AUTHORIZATION_FA.md](PILOT_EXECUTION_AUTHORIZATION_FA.md): گیت و بودجه مصوب
- [RECOVERY_CHECKPOINT_FA.md](RECOVERY_CHECKPOINT_FA.md): گزارش شکست واقعی پس از fit و بازیابی بی‌آموزش
- [results/pilot_seed20261102_shared_two_fits/](results/pilot_seed20261102_shared_two_fits/): raw، fingerprint، hash، trace، coverage، کامل‌ترین metric report، CSV و SVG
- [tables/pilot_summary.tex](tables/pilot_summary.tex) و [figures/validation_ba.svg](figures/validation_ba.svg): منابع مشتق‌شده از همان raw

## اجرای مجدد بدون آموزش برای QA
```bash
python -m unittest discover -v -s code/experiments/ch4_loss_controlled_comparison -p 'test_*.py'
python -m unittest discover -v -s code/experiments/benchmark_v1 -p 'test_common_metrics_contract.py'
python -m unittest discover -v -s code/experiments/benchmark_v1 -p 'test_snapshot_pipeline.py'
python -m unittest discover -v -s code/experiments/benchmark_v1 -p 'test_snapshot_roundtrip.py'
```
`finalize_existing.py` تنها raw موجود را بازخوانی و خروجی دیجیتال را از آن بازسازی می‌کند؛ در آن optimizer وجود ندارد. `run_experiment.py` به‌طور پیش‌فرض dry-run و در حالت --execute fail-closed است؛ با پایان بودجه دو fit این مرحله مجوز تکرار آن وجود ندارد.

**درخواست گیت:** `PILOT_READY_FOR_QA`؛ هیچ پذیرش نتیجه علمی یا مجوز مرحله اصلی پیش از رأی ممیزی مستقل.
