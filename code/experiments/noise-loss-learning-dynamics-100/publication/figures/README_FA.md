# نمودارهای هندسهٔ Train و عملکرد Validation — §۴.۱۴

**منشأ:** ۱۵۲ آموزش موجود در [Run #44419173443](https://github.com/mahhammad/My-Thesis/actions/runs/44419173443)، بدون اجرای دوباره. برای مقایسه صرفاً |A_N|=1,...,6 و شش زیرمجموعهٔ وابستهٔ هر سطح استفاده شده است.

در هر شکل، دو تابع زیان Sigmoid-JS و Softplus-JS با میانگین پررنگ و مسیرهای منفرد کم‌رنگ نمایش داده می‌شوند. این مقایسهٔ **مقدار سنجهٔ یکسان** است، نه مقایسهٔ مقدار خام دو تابع Loss؛ نمودارهای Loss و گرادیان به‌درستی [جدا هستند](../loss-gradient/README_FA.md).

## شکل‌های برداری برای پایان‌نامه

| معیار | سیاست مشترک | سیاست وابسته به کلاس |
|---|---|---|
| جدایی مراکز I، فقط Train | [SVG](shared/separation_i.svg) | [SVG](class/separation_i.svg) |
| تراکم R_JS، فقط Train | [SVG](shared/density_r_js.svg) | [SVG](class/density_r_js.svg) |
| تراکم R_KL، فقط Train | [SVG](shared/density_r_kl.svg) | [SVG](class/density_r_kl.svg) |
| تراکم R_sqrt_JS، فقط Train | [SVG](shared/density_r_sqrt_js.svg) | [SVG](class/density_r_sqrt_js.svg) |
| معیار T، فقط Train | [SVG](shared/separation_t.svg) | [SVG](class/separation_t.svg) |
| معیار BT، فقط Train | [SVG](shared/separation_bt.svg) | [SVG](class/separation_bt.svg) |
| معیار TNL، فقط Train | [SVG](shared/separation_tnl.svg) | [SVG](class/separation_tnl.svg) |
| معیار BTNL، فقط Train | [SVG](shared/separation_btnl.svg) | [SVG](class/separation_btnl.svg) |
| میانگین حاشیه، فقط Train | [SVG](shared/margin_mean.svg) | [SVG](class/margin_mean.svg) |
| Accuracy آموزش | [SVG](shared/accuracy_train.svg) | [SVG](class/accuracy_train.svg) |
| BA آموزش | [SVG](shared/balanced_accuracy_train.svg) | [SVG](class/balanced_accuracy_train.svg) |
| BA اعتبارسنجی | [SVG](shared/balanced_accuracy_validation.svg) | [SVG](class/balanced_accuracy_validation.svg) |
| Accuracy اعتبارسنجی | [SVG](shared/accuracy_validation.svg) | [SVG](class/accuracy_validation.svg) |
| Precision اعتبارسنجی | [SVG](shared/precision_macro_validation.svg) | [SVG](class/precision_macro_validation.svg) |
| Recall اعتبارسنجی | [SVG](shared/recall_macro_validation.svg) | [SVG](class/recall_macro_validation.svg) |

تمام شکل‌ها با همین نام در PNG نیز موجودند. محورهای هر گروه از شش پنل، مقیاس رُند مشترک و متناسب با داده دارند. خط‌چین نمودارهای Accuracy و BA، مرجع بیزی MAP همان تقسیم داده است (جدول `tab:ch4-base-bayes-split` در فصل چهارم). محور افقی، گام‌های ۰ تا ۱۰۰ را نشان می‌دهد.

**مرز استنباط:** سنجه‌های I و R تنها برای Train محاسبه شده‌اند؛ Validation در هندسهٔ مراکز، Loss، گرادیان یا آموزش دخالت ندارد. شش زیرمجموعه از یک دادهٔ مادر هستند، نه شش تکرار مستقل.
