# بازتولید فنی آزمایش S-103 و گواه مستقل

## محل داده خام
دو ZIP مالک در Library زیر مسیر `/My-Thesis/Section-4-11/`:
- `section411_standard_S_103_audit_package.zip`: SHA256 `742b38249fccf904fc5bf6e783a4110853a1a7051433f7b8417b32c1774493f4`.
- `V_R02_SEC411_INDEPENDENT_QA_EVIDENCE.zip`: SHA256 `1d6c42e297b4c00c654bb4dec96d7f9d114240b3fb56974353968f0ade96c086`.

**این دو ZIP در GitHub tree فعلی کپی نشده‌اند.** در مکانی مطمئن آن‌ها را قرار دهید؛ قبل از اعتماد به فایل‌ها، SHA کامل را بازحساب کنید.

```bash
sha256sum section411_standard_S_103_audit_package.zip V_R02_SEC411_INDEPENDENT_QA_EVIDENCE.zip
python - <<'PY'
import zipfile, hashlib
with zipfile.ZipFile('section411_standard_S_103_audit_package.zip') as z:
    assert z.testzip() is None
print('ZIP integrity: PASS')
PY
```

بسته اول: `README_FIRST_FA.md`، `REPRODUCE_FA.md`، `SHA256SUMS.txt`، `data/freeze_manifest.json`، `results/SEC411_STANDARD_S_103_SUMMARY.json`، کدهای `src/`، ۵۴ کران `results/bounds_*.json`، شش `checks/precision_*.json`، ۳۶ `results/solver_*.json`، شش `results/replay_*.json` و یک پروب منابع را شامل می‌شود. تمام ۱۶۱ فایلِ مشمول manifest در رأی ممیز بازهش شدند.

بسته دوم: کد مستقل `rigorous_directed_bellman.py` و `independent_pinsker_fraction.py`، شش محصورسازی `directed_baseline_*.json`، فایل `independent_pinsker_certificates.json`، گزارش و برهان مستقل. ۲۱ فایل در manifest مستقل قفل شده‌اند. این پیاده‌سازی از V2 و محصورسازی نویسنده import نمی‌کند.

برای بازتولید بدون محاسبه سنگین، فقط JSON/CSV خام و این گزارش را بخوانید؛ اجرای solver یا شبکه کامل تنها با پروتکل و منابع مصوب صورت گیرد. دستور تولید دوباره و تمام dependencyها در `REPRODUCE_FA.md` داخل بسته اصلی وجود دارند.

## کنترل سبک جدول‌های جدید
```bash
python code/experiments/bayes_center_reachability_standard/src/verify_presentation.py
```

این کنترل از `results/certified_summary.json`، `results/metric_seed_summary.json` و فایل‌های `tables/` استفاده می‌کند. به داده خام یا تغییر نتایج نیاز ندارد. کنترل کامل منبع raw در `tests/qa_tests.py` داخل بسته اول و کد مستقل ممیز در بسته دوم انجام می‌شود.

**محدودیت:** موفقیت تست فایل‌های خلاصه اثبات نمی‌کند محصورسازی float برای ورودی نامحدود درست است. گواه محدود شش‌گانه به بررسی مستقل ممیز و ورودی‌های منجمد وابسته است.
