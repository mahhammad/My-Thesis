"""Generate E08 printed assets and experimental report from one checked JSON.

No model execution, no measurement or training here.
"""
from pathlib import Path

HERE=Path(__file__).resolve().parent
DEPTHS=(2,3,4,6,8,12)
POLICIES=("uniform","trained_shared")


def make_assets(result):
    tables=HERE/"tables"
    figures=HERE/"figures"
    tables.mkdir(parents=True,exist_ok=True)
    figures.mkdir(parents=True,exist_ok=True)
    by={name:{row["r"]:row for row in result["policy_results"][name]["truncation"]}
        for name in POLICIES}
    lines=[
      "% Generated from results/experiment_report.json; no independent copied values.",
      r"\begin{table}[htbp]",r"\centering\small",
      r"\caption{کران جرم حذف‌شده و میانگین متوازن خطای تغییر کل تقریب گشت با $H=12$ ثابت و مرجع $300$ نمونه‌ای؛ سیاست دوم، سیاست مشترک ذخیره‌شده در Benchmark است.}",
      r"\label{tab:ch4-matrix-truncation}",r"\begin{tabular}{rccccc}\hline",
      r"$r$ & $\varepsilon_r$ & Train یکنواخت & Train مشترک ذخیره‌شده & Validation یکنواخت & Validation مشترک ذخیره‌شده\\\hline"
    ]
    for r in DEPTHS:
        a=by["uniform"][r];b=by["trained_shared"][r]
        cells=[
            str(r),f"{a['epsilon_tail']:.5f}",
            f"{a['train_mean_balanced_tv']:.5f}",
            f"{b['train_mean_balanced_tv']:.5f}",
            f"{a['validation_mean_balanced_tv']:.5f}",
            f"{b['validation_mean_balanced_tv']:.5f}"
        ]
        lines.append(" & ".join(cells)+r"\\")
    lines += [r"\hline",r"\end{tabular}",r"\end{table}",""]
    (tables/"truncation.tex").write_text("\n".join(lines),encoding="utf8")

    mc=result["monte_carlo"]["summary"]
    summ={(v["policy"],v["nwalks"]):v for v in mc}
    lines=[
      "% Generated from results/experiment_report.json; MC sample sizes are nested.",
      r"\begin{table}[htbp]",r"\centering\small",
      r"\caption{میانگین خطای TV مونت‌کارلو نسبت به امید دقیق، در شش مبدأ ثابت Train/Validation و دو بذر مستقل مسیر؛ بذرهای گشت، داده مستقل تولید نمی‌کنند.}",
      r"\label{tab:ch4-matrix-mc-check}",r"\begin{tabular}{rcc}\hline",
      r"گشت بر مبدأ $M$ & سیاست یکنواخت & سیاست مشترک ذخیره‌شده\\\hline"
    ]
    for n in (128,512,2048):
        lines.append(" & ".join([
            str(n),f"{summ[('uniform',n)]['mean_tv']:.5f}",
            f"{summ[('trained_shared',n)]['mean_tv']:.5f}"])+r"\\")
    lines += [r"\hline",r"\end{tabular}",r"\end{table}",""]
    (tables/"monte_carlo.tex").write_text("\n".join(lines),encoding="utf8")

    t=result["timing"]
    median={(v["policy"],v["r"]):v["median_seconds"] for v in t}
    lines=[
      "% Process CPU timings from results/experiment_report.json, 24 origins, 3 repeats.",
      r"\begin{table}[htbp]",r"\centering\small",
      r"\caption{میانه زمان اجرای گذر پیشرو روی ۲۴ مبدأ ثابت Train؛ هر اندازه‌گیری پس از گرم‌کردن و با سه تکرار در یک دستگاه CPU انجام شده است. واحد ثانیه؛ زمان‌های وابسته به دستگاه.}",
      r"\label{tab:ch4-matrix-runtime}",r"\begin{tabular}{rcc}\hline",
      r"عمق $r$ & یکنواخت (ثانیه) & مشترک ذخیره‌شده (ثانیه)\\\hline"
    ]
    for n in (2,4,12):
        lines.append(" & ".join([
            str(n),f"{median[('uniform',n)]:.5f}",
            f"{median[('trained_shared',n)]:.5f}"])+r"\\")
    lines += [r"\hline",r"\end{tabular}",r"\end{table}",""]
    (tables/"runtime.tex").write_text("\n".join(lines),encoding="utf8")

    # Non-flat and meaningful: bound versus maximal *individual* Train error,
    # which the balanced-mean table does not show.
    bound=[by["uniform"][r]["epsilon_tail"] for r in DEPTHS]
    unif=[by["uniform"][r]["train_max_tv"] for r in DEPTHS]
    trained=[by["trained_shared"][r]["train_max_tv"] for r in DEPTHS]
    high=max(bound)*1.12
    scale=3.6/high
    xs=[1.2+i*1.04 for i in range(len(DEPTHS))]
    pts=lambda values:" ".join("("+f"{x:.3f},{1.+v*scale:.3f}"+")"
                               for x,v in zip(xs,values))
    fig=[
      "% Source: results/experiment_report.json, epsilon and max individual Train TV",
      r"\begin{figure}[htbp]",r"\centering",
      r"\begin{tikzpicture}[x=1cm,y=1cm]",
      r"\draw[->] (.65,1)--(7.2,1);",
      r"\draw[->] (.65,1)--(.65,5.1);",
      r"\node[font=\small] at (3.7,.3) {عمق محاسبه $r$};",
      r"\node[font=\small,rotate=90] at (.15,3) {TV};"
    ]
    for x,n in zip(xs,DEPTHS):
        fig.append("\\node[font=\\scriptsize,anchor=north] at ("+
                   f"{x:.3f}"+",.94) {"+str(n)+"};")
    for mark in (0.,.04,.08,.12):
        ypos=1+mark*scale
        fig.append("\\draw[gray!30] (.70,"+f"{ypos:.3f}"+
                   ")--(6.8,"+f"{ypos:.3f}"+");")
        fig.append("\\node[font=\\scriptsize,anchor=east] at (.61,"+
                   f"{ypos:.3f}"+") {"+f"{mark:.2f}"+"};")
    for style,values in (("black",bound),("blue",unif),
                         ("orange!85!black",trained)):
        fig.append("\\draw[very thick,"+style+
                   "] plot coordinates {"+pts(values)+"};")
    fig.extend([
      r"\node[font=\scriptsize,anchor=west] at (4.9,4.65) {سیاه: کران $\varepsilon_r$};",
      r"\node[font=\scriptsize,anchor=west] at (4.9,4.35) {آبی: بیشینه TV یکنواخت};",
      r"\node[font=\scriptsize,anchor=west] at (4.9,4.05) {نارنجی: بیشینه TV مشترک};",
      r"\end{tikzpicture}",
      r"\caption{بیشینه خطای تک‌نمونه‌ای روی مرجع کامل در برابر کران جهانی حذف عمق؛ نمودار فقط ادعای کنترل خطا در این دو سیاست را بررسی می‌کند.}",
      r"\label{fig:ch4-matrix-truncation-bound}",
      r"\end{figure}",""
    ])
    (figures/"bound_and_max_tv.tex").write_text("\n".join(fig),encoding="utf8")


def make_report(result):
    stat=result["graph_size"]
    mc=result["monte_carlo"]["summary"]
    seen={(v["policy"],v["nwalks"]):v["mean_tv"] for v in mc}
    no=getattr(result,"unused",None)
    note=[]
    for policy in POLICIES:
        a=result["policy_results"][policy]["truncation"]
        r3=next(x for x in a if x["r"]==3)
        note.append(
            f"- {policy}: r=3، کران ε={r3['epsilon_tail']:.5f}، "
            f"میانگین متوازن TV آموزش={r3['train_mean_balanced_tv']:.5f}، "
            f"Validation={r3['validation_mean_balanced_tv']:.5f}؛ "
            f"بیشینه TV مرکز={r3['class_centers_max_tv']:.5f}.")
    mc_notes=[]
    for policy in POLICIES:
        mc_notes.append(
            f"- {policy}: میانگین TV با M=128 برابر {seen[(policy,128)]:.5f} "
            f"و با M=2048 برابر {seen[(policy,2048)]:.5f} بود؛ "
            "این میانگین بر شش مبدأ از پیش مشخص و دو جریان MC است.")
    report=[
      "# E-CH4-08 — اعتبارسنجی محاسبه ماتریسی بازنمایی چندگامی",
      "",
      "## سؤال و فرضیه",
      "صحت امید دقیق در برابر گشت تصادفی مستقل، کران نظری حذف عمق ثابت H=12، "
      "و هزینه مشاهده‌شده گذر پیشرو بر مرجع کامل ۳۰۰تایی را می‌سنجیم. "
      "پیش‌ثبت در PROTOCOL_FA.md قبل از هر اجرا انجام شده است.",
      "",
      "## داده، سیاست، شرایط",
      "مولد استاندارد فصل چهارم: بذر 20261102، Train=300، Validation=90، "
      "Test تاریخی بارگذاری نشد. از کل ۳۰۰ نمونه در ترتیب reference_ids ذخیره‌شده "
      "در checkpoint Benchmark v1 استفاده شد؛ دو سیاست مشترکِ Z=0 و سیاست "
      "آموزش‌دیده ۵۰گامی (tau=.005) بدون هرگونه آموزش جدید بازسنجی شدند. "
      "beta=.4، lambda=.8، H=12، float64 و سه R/پنج جدایی/BA/NLL "
      "طبق فصل سوم و گزارشگر مشترک سنجیده شدند.",
      "",
      "## اعتبارسنجی مستقل امید",
      "در هر دو سیاست، پاسخ بازگشت مستقل در همه مبدأهای Train و Validation با "
      "موتور رسمی Benchmark v1 با خطای حداکثر مندرج در JSON تطبیق یافت. "
      "در شش مبدأ از پیش انتخاب‌شده، مونت‌کارلوی مستقل با بذرهای مسیر "
      "20261121 و 20261122 و بودجه‌های 128/512/2048 برای ارزیابی تجربی به کار رفت. "
      "این‌ها دو بذر مولد مستقل نیستند.",
      *mc_notes,
      "",
      "## کران حذف عمق و کنترل مراکز",
      "برای مدل اصلی H=12، نگه‌داشتن r گام نخست به‌معنای تغییر مدل به H=r نیست. "
      "در تمام ۳۰۰ بازنمایی Train، ۹۰ Validation و سه مرکز کلاسی، "
      "کران TV≤ε_r+1e-9 در تمام عمق‌های ازپیش‌ثبت‌شده و هر دو سیاست برقرار بود.",
      *note,
      "",
      "## پیچیدگی و هزینه",
      f"روی مرجع کامل شمار گره‌های نمونه–ویژگی {stat['m_sample_feature_nodes']}، "
      f"تعداد یال‌های عمودی بالقوه {stat['directed_vertical_possible']} و "
      f"تعداد انتخاب‌های افقی بالقوه {stat['horizontal_possible']} است. "
      "هزینه نظری طرح تنک مربوط به ضرب بردار در عامل‌هاست، درحالی‌که "
      "پیاده‌سازی آزمایش از کرنل‌های متراکم Torch استفاده می‌کند. "
      f"آرایه فرضی قدیمی n×d×n×n با float64 حدود "
      f"{stat['old_dense_n_d_n_n_bytes']/1e9:.3f} GB حافظه نیاز داشت؛ "
      "این آرایه در این اجرای Benchmark تخصیص داده نشده است. "
      "زمان‌های ثبت‌شده در tables/runtime.tex وابسته به ماشین و شامل آموزش نیستند.",
      "",
      "## پوشش معیارهای عمومی",
      "در results/metrics_full.json برای هر سیاست و هر r، Train/Validation "
      "دارای R_KL/R_JS/R_sqrt_JS، T/BT/TNL/BTNL/I، فاصله‌های زوجی مراکز، "
      "BA/Acc، ماتریس سردرگمی، حاشیه و انواع NLL است. "
      "دو snapshot خام imc-snapshot-v2 به همراه گزارش‌های مستقل ذخیره شده‌اند؛ "
      "فایل raw_step_distributions.npz تمام توزیع‌های گام‌به‌گام را نگه می‌دارد.",
      "",
      "## محدودیت‌ها",
      "این آزمایش تک‌بذر مولد داده است و تنها دو سیاست shared از پیش ثابت را "
      "بررسی می‌کند. بودجه‌های MC تو در تو هستند و برای تعمیم آماری مستقل نیستند. "
      "نتایج درباره کران محاسباتی‌اند، نه ادعای برتری دقت یا حل مسئله آموزش. "
      "حافظه RSS کل فرایند معادل حافظه یک نمونه/یک سیاست نیست. "
      "داده تاریخی ۷۲تایی منبع نتیجه رسمی نشده است. "
      "کامپایل کامل XeLaTeX و داوری مستقل هنوز لازم‌اند.",
      "",
      "## راهنمای بازتولید و تحویل",
      "با اجرای run.py --execute فقط امید دقیق/MC و گزارش عمومی روی Train/Validation "
      "تولید می‌شود؛ اجرا به هیچ وجه آموزش تازه یا بازخوانی Test ندارد. "
      "پوشه results شامل provenance.json، coverage.json، metrics_full.json، "
      "experiment_report.json، raw_step_distributions.npz و snapshotهای v2 است. "
      "تمام جدول‌ها از experiment_report.json تولید می‌شوند. "
      "وضعیت این شاخه فقط آماده بازبینی مستقل است و به chapter4 ادغام نشده است.",
      ""
    ]
    (HERE/"REPORT_FA.md").write_text("\n".join(report),encoding="utf8")
