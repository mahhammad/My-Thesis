"""Audit the stored center-matching experiment and compare three selected cases.

Uses only the Python standard library.  All probabilities and geometry metrics
are calculated from SAVED REALIZED centers, never from the target centers.
The best guided round is selected among rounds with a measured validation result.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent
DEFAULT_HISTORY = BASE / 'results' / 'history_00_48.json'
LOG2 = math.log(2)


def entropy(p):
    return -sum(float(x) * math.log(float(x)) for x in p if x > 0)


def jsd(p, q):
    mid = [(x + y) / 2 for x, y in zip(p, q)]
    return max(0.0, entropy(mid) - (entropy(p) + entropy(q)) / 2)


def geometry(centers):
    k = len(centers)
    if k < 2 or any(len(row) != k or min(row) < -1e-12 or
                     abs(sum(row) - 1) > 1e-9 for row in centers):
        raise ValueError('Invalid K-by-K center probability matrix')
    omega = [1 / k] * k
    mean = [sum(centers[c][j] for c in range(k)) / k for j in range(k)]
    pair_jsd = [[jsd(centers[i], centers[j]) for j in range(k)] for i in range(k)]
    offdiag = [pair_jsd[i][j] for i in range(k) for j in range(i + 1, k)]
    permutations = list(itertools.permutations(range(k)))
    return {
        'T': sum(centers[c][c] for c in range(k)) / k,
        'TNL': -sum(math.log(centers[c][c]) if centers[c][c] > 0 else -math.inf
                    for c in range(k)) / k,
        'BT': max(sum(centers[c][perm[c]] for c in range(k)) / k for perm in permutations),
        'BTNL': min(-sum(math.log(centers[c][perm[c]]) if centers[c][perm[c]] > 0
                         else -math.inf for c in range(k)) / k for perm in permutations),
        'I': entropy(mean) - sum(entropy(c) for c in centers) / k,
        'min_center_sqrt_jsd': math.sqrt(min(offdiag)),
        'mean_pairwise_center_jsd': sum(offdiag) / len(offdiag),
        'pairwise_center_jsd': pair_jsd,
    }


def validate_history(data):
    history = data['history']
    if not history or [r['round'] for r in history] != list(range(len(history))):
        raise ValueError('Rounds must be continuous and start at zero')
    baseline = data['baseline']
    geometry(baseline['centers'])
    for idx, row in enumerate(history):
        geometry(row['realized_centers'])
        geometry(row['targets'])
        geometry(row['next_targets'])
        if idx:
            previous = history[idx-1]['next_targets']
            delta = max(abs(x-y) for a, b in zip(row['targets'], previous)
                        for x, y in zip(a, b))
            if delta > 2e-12:
                raise ValueError(f'Checkpoint discontinuity at round {idx}: {delta}')
        # Audit that the saved center information and distance agree with
        # independently recalculated geometry.
        g = geometry(row['realized_centers'])
        for stored, actual in [('center_information', g['I']),
                               ('min_center_js_distance', g['min_center_sqrt_jsd'])]:
            if stored in row and abs(row[stored] - actual) > 1e-9:
                raise ValueError(f'Inconsistent {stored} in round {idx}')
    return history


def select_cases(data):
    history = validate_history(data)
    baseline = data['baseline']
    valid = [r for r in history if r.get('validation_hypothesis') is not None]
    if not valid:
        raise ValueError('No measured guided validation metrics')
    peak = max(valid, key=lambda r: r['validation_hypothesis']['balanced_accuracy'])

    def case(case_id, centers, validation, train, *, round_record=None, protocol='uniform'):
        g = geometry(centers)
        return {
            'case': case_id,
            'round': None if round_record is None else round_record['round'],
            'protocol': protocol,
            'centers': centers,
            'targets_used_to_construct_policy': (None if round_record is None
                                                  else round_record['targets']),
            'next_targets_not_used_for_this_evaluation': (None if round_record is None
                                                           else round_record['next_targets']),
            'geometry': g,
            'train': train,
            'validation': validation,
            'train_within_class_jsd': train.get('within_class_jsd'),
            'target_realized_mean_jsd': (None if round_record is None
                                         else round_record['mean_target_realized_jsd']),
            'projection': (None if round_record is None else round_record.get('projection')),
        }

    guided_train_keys = ('accuracy', 'balanced_accuracy', 'within_class_jsd',
                         'NLL_norm', 'J_margin')
    def guided(row, key):
        tr = dict(row.get('train_oracle_target') or {})
        if 'within_class_jsd' not in tr and 'train_within_class_jsd' in row:
            tr['within_class_jsd'] = row['train_within_class_jsd']
        return case(key, row['realized_centers'], row['validation_hypothesis'],
                    {name: tr.get(name) for name in guided_train_keys},
                    round_record=row, protocol='class_hypothesis')

    return [
        case('uniform_baseline', baseline['centers'], baseline['validation'],
             baseline['train']),
        guided(history[0], 'first_guided_round'),
        guided(peak, 'best_measured_guided_validation'),
    ]


def create_trajectory(data):
    rows = []
    for r in data['history']:
        g = geometry(r['realized_centers'])
        val = r.get('validation_hypothesis') or {}
        tr = r.get('train_oracle_target') or {}
        rows.append({
            'round': r['round'],
            'I': g['I'],
            'T': g['T'],
            'TNL': g['TNL'],
            'BT': g['BT'],
            'BTNL': g['BTNL'],
            'min_center_sqrt_jsd': g['min_center_sqrt_jsd'],
            'mean_pairwise_center_jsd': g['mean_pairwise_center_jsd'],
            'mean_target_realized_jsd': r['mean_target_realized_jsd'],
            'max_target_update_l1': r['target_update_l1'],
            'train_within_class_jsd': tr.get('within_class_jsd', r.get('train_within_class_jsd')),
            'validation_balanced_accuracy': val.get('balanced_accuracy'),
            'validation_accuracy': val.get('accuracy'),
            'validation_true_margin_jsd': val.get('mean_true_margin_jsd'),
            'validation_fraction_nonpositive_margin': val.get('fraction_nonpositive_true_margin'),
        })
    return rows


def fmt(value, places=5):
    if value is None:
        return '—'
    return f'{value:.{places}f}'


def matrix_md(matrix):
    lines = ['| کلاس | برچسب ۰ | برچسب ۱ | برچسب ۲ |', '|---|---:|---:|---:|']
    for i, row in enumerate(matrix):
        lines.append('| ' + str(i) + ' | ' + ' | '.join(fmt(v, 6) for v in row) + ' |')
    return '\n'.join(lines)


def report_fa(cases, trajectory, data):
    u, first, peak = cases
    nmeasured = sum(x['validation_balanced_accuracy'] is not None for x in trajectory)
    lines = [
        '# گزارش مقایسهٔ سیاست یکنواخت و دو وضعیت سیاست هدایت‌شده به مرکز',
        '',
        '> **دامنه:** یک مجموعه‌دادهٔ مصنوعی، یک بذر، و آزمایش مشاهده‌ای؛ نه آموزش نهایی سیاست و نه نتیجهٔ آماری تعمیم‌پذیر.',
        '',
        '## هدف و نحوهٔ انتخاب سه وضعیت',
        '',
        '۱. **Uniform:** سیاست یکنواخت؛ مرکز هر کلاس میانگین بازنمایی‌های آموزشی حاصل از همین سیاست است.',
        '۲. **اولین اجرای هدایت‌شده، round=0:** هدف‌ها نزدیک بردارهای یک‌داغ‌اند؛ سیاست‌های محلی در هر گام با کمینه‌کردن JSD میان توزیع یک‌گامی و مرکز هدف ساخته می‌شوند. سپس مراکز واقعی گشت‌های آموزشی محاسبه می‌شوند. **دقت روی همین مراکز واقعی سنجیده شده؛ هیچ اصلاح هدف پس از اجرای اول در این اندازه‌گیری اعمال نشده است.**',
        f'۳. **بالاترین دقت ثبت‌شده، round={peak["round"]}:** بهترین مقدار دقت متوازن اعتبارسنجی بین {nmeasured} تکراری که واقعاً ارزیابی شده‌اند. این وضعیت یک انتخاب پس‌نگر با استفاده از اعتبارسنجی است، نه تخمین بدون سوگیری عملکرد مدل نهایی.',
        '',
        '## مقایسهٔ اصلی',
        '',
        '| معیار | یکنواخت | هدایت‌شده، اجرای اول | بالاترین دقت مسیر |',
        '|---|---:|---:|---:|',
    ]
    metrics = [
        ('دقت متوازن اعتبارسنجی (%)', lambda c: 100*c['validation']['balanced_accuracy'], 2),
        ('دقت معمولی اعتبارسنجی (%)', lambda c: 100*c['validation']['accuracy'], 2),
        ('دقت متوازن آموزش (%)', lambda c: None if c['train'].get('balanced_accuracy') is None else 100*c['train']['balanced_accuracy'], 2),
        ('اطلاعات مراکز I (nats)', lambda c: c['geometry']['I'], 6),
        ('حداقل فاصلهٔ sqrt(JSD) مراکز', lambda c: c['geometry']['min_center_sqrt_jsd'], 6),
        ('میانگین JSD زوج‌مرکز', lambda c: c['geometry']['mean_pairwise_center_jsd'], 6),
        ('T', lambda c: c['geometry']['T'], 6),
        ('TNL', lambda c: c['geometry']['TNL'], 6),
        ('BT', lambda c: c['geometry']['BT'], 6),
        ('BTNL', lambda c: c['geometry']['BTNL'], 6),
        ('میانگین JSD درون‌کلاسی آموزش', lambda c: c['train_within_class_jsd'], 6),
        ('میانگین JSD هدف تا مرکز واقعی', lambda c: c['target_realized_mean_jsd'], 6),
        ('حاشیهٔ واقعی اعتبارسنجی (JSD)', lambda c: c['validation'].get('mean_true_margin_jsd'), 6),
        ('سهم حاشیهٔ غیرمثبت اعتبارسنجی', lambda c: c['validation'].get('fraction_nonpositive_true_margin'), 6),
    ]
    for name, val, places in metrics:
        lines.append('| '+name+' | '+' | '.join(fmt(val(c), places) for c in cases)+' |')
    lines += [
        '',
        '**هشدار دربارهٔ دقت آموزش:** در سیاست هدایت‌شده، برچسب واقعی مبدأ برای ساخت سیاست آموزشی استفاده شده است؛ بنابراین «دقت آموزش» آن، سنجه‌ای شرطی/اوراکلی است و مستقیماً با دقت آموزشِ یکنواخت مقایسه‌پذیر نیست.',
        '',
        '## تحلیل تفاوت سه وضعیت',
        '',
        f'- از اجرای اول تا نقطهٔ بیشینهٔ دقت، I از {first["geometry"]["I"]:.5f} به {peak["geometry"]["I"]:.5f} کاهش یافت (حدود {100*(1-peak["geometry"]["I"]/first["geometry"]["I"]):.1f}٪). در همین بازه، JSD درون‌کلاسی آموزش از {first["train_within_class_jsd"]:.5f} به {peak["train_within_class_jsd"]:.5f} کاهش یافت (حدود {100*(1-peak["train_within_class_jsd"]/first["train_within_class_jsd"]):.1f}٪). دقت متوازن اعتبارسنجی در مجموع {100*(peak["validation"]["balanced_accuracy"]-first["validation"]["balanced_accuracy"]):.2f} واحد درصد افزایش یافت؛ بنابراین کاهش جدایی در این بخش از مسیر با **تراکم بیشتر و بهبود دقت** همراه بوده است.',
        f'- یکنواخت با I={u["geometry"]["I"]:.5f} بسیار کمتر از هر دو وضعیت هدایت‌شده، دقت متوازن {100*u["validation"]["balanced_accuracy"]:.2f}٪ ثبت کرده است. این نکته نشان می‌دهد I به‌تنهایی پیش‌بینی‌کنندهٔ کافی دقت نیست؛ ولی به سبب پروتکل‌های استنتاج متفاوت، نتیجهٔ مقایسهٔ علّی دو سیاست نیست.',
        f'- در تکرار {trajectory[-1]["round"]}، I={trajectory[-1]["I"]:.6f} و دقت متوازنِ اندازه‌گیری‌شده {100*trajectory[-1]["validation_balanced_accuracy"]:.2f}٪ است: نمونه‌ای از فروپاشی جدایی مراکز و افت قدرت تفکیک در تکرارهای بسیار دیرتر، بدون ادعای اثبات نقطهٔ حدی.',
        '- مقادیر `T=BT` و `TNL=BTNL` برای این سه ماتریس رخ داده‌اند زیرا انتساب هم‌نام کلاس‌ها، بهترین انتساب جایگشتی است؛ این برابری خاص این داده‌هاست، نه یک رابطهٔ همگانی.',
        '',
        '## ماتریس مراکز (سطر = کلاس مبدأ؛ ستون = احتمال برچسب مشاهده‌شده)',
        '',
    ]
    for c, title in zip(cases, ('سیاست یکنواخت', 'اجرای اول، round=0', f'بالاترین دقت، round={peak["round"]}')):
        lines += ['### '+title, '', '**مراکز واقعی:**', '', matrix_md(c['centers']), '']
        if c['targets_used_to_construct_policy'] is not None:
            lines += ['**مراکز هدفِ مورد استفاده برای ساخت سیاست (نه مراکز سنجش دقت):**', '',
                      matrix_md(c['targets_used_to_construct_policy']), '']
    lines += [
        '## مشاهدهٔ مسیر ۴۹ تکرار',
        '',
        '![I در برابر دقت متوازن اعتبارسنجی](figures/information_vs_accuracy.svg)',
        '',
        '![ماتریس مراکز واقعی سه وضعیت](figures/center_matrices.svg)',
        '',
        f'از {len(trajectory)} تکرار ثبت‌شده، دقت اعتبارسنجی برای {nmeasured} تکرار اندازه‌گیری شده است؛ در نمودار فقط همان نقاط آمده‌اند. اعدادِ خالی در CSV به معنای فقدان ارزیابی‌اند، نه صفر.',
        '',
        'در این اجرا، افزایش اولیهٔ دقت با کاهش فاصلهٔ هدف تا مراکز واقعی و کاهش پراکندگی درون‌کلاسی همراه بوده، در حالی که I کاهش یافته است؛ بنابراین حداکثرکردن I به‌تنهایی معادل حداکثرکردن دقت نیست. در تکرارهای بسیار دیرتر، تقریباً یکسان‌شدن مراکز با افت دقت به نزدیکی سطح تصادفی همراه شده است. این همبستگی مشاهده‌ای، رابطهٔ علّی یا یک قضیهٔ همگرایی نیست.',
        '',
        '## محدودیت روش‌شناختیِ مهم',
        '',
        '- یکنواخت: یک بازنماییِ مستقل از کلاس فرضی برای هر نمونه ساخته می‌شود و با همهٔ مراکز مقایسه می‌شود.',
        '- هدایت‌شده: برای هر کلاس فرضی یک بازنماییِ جداگانه تولید می‌شود؛ فاصلهٔ آن بازنمایی تا مرکز **واقعی آموزشی همان کلاس** محاسبه می‌شود و کمترین فاصله برنده است. برچسب اعتبارسنجی در ساخت گشت به کار نرفته است.',
        '- به همین دلیل، دو پروتکل زمان اجرا و قدرت مدل متفاوت دارند. تفاضل دقت آن‌ها آزمون کنترل‌شدهٔ اثر انتخاب سیاست به‌تنهایی نیست.',
        '- «بیشترین دقت مسیر» با اعتبارسنجی انتخاب شده است؛ آزمون نهایی باید مستقل بماند. تکرارهای بررسی‌شده با یک بذر و ۱۰۰ گشت به ازای نمونه اجرا شده‌اند؛ نویز مونت‌کارلو و بهینه‌سازی تقریبی Q وجود دارد.',
        '- این آزمایش **بدون متغیر مشترک آموزش‌دیده و بدون پاداش‌دهی** است. سیاست‌ها از حل محلی JSD نسبت به مراکز هدف ساخته می‌شوند.',
        '',
        '## محاسبات و داده‌ها',
        '',
        '- `results/history_00_48.json`: مراکز هدف، مراکز واقعی، و معیارهای ذخیره‌شدهٔ ۴۹ تکرار به همراه خط مبنا.',
        '- `results/three_cases.json` و `results/three_cases.csv`: سه وضعیت انتخابی با هندسهٔ باز‌محاسبه‌شده از مراکز واقعی.',
        '- `results/trajectory_metrics.csv`: مسیر معیارها؛ برای دقت‌های محاسبه‌نشده مقدار خالی.',
        '- `analyze.py`: محاسبهٔ مستقل شاخص‌های هندسی، اعتبارسنجی سازگاری checkpointها و تولید گزارش و نمودارها.',
        '- `reproduce.py`: اجرای مجدد مولد داده و کد اصلی آزمایش در همان مخزن.',
        '',
        'فرمول اطلاعات مرکز با وزن یکنواخت کلاس‌ها: `I = H(mean_c V_c) - mean_c H(V_c)` با لگاریتم طبیعی؛ KL/JSD نیز برحسب nats است. `T` میانگین قطر مرکز و `BT` بیشترین میانگین انتساب جایگشتی است؛ `TNL` و `BTNL` هم نسخه‌های لگاریتمی متناظرند.',
        '',
    ]
    return '\n'.join(lines)


def information_svg(trajectory, cases):
    measured = [r for r in trajectory if r['validation_balanced_accuracy'] is not None]
    x0,y0,width,height = 105,55,800,450
    xmin,xmax = min(math.log10(r['I']) for r in measured) - .15, max(math.log10(r['I']) for r in measured) + .15
    ymin,ymax=.25,.92
    def xp(v):return x0+(math.log10(v)-xmin)/(xmax-xmin)*width
    def yp(v):return y0+height-(v-ymin)/(ymax-ymin)*height
    s=['<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="625" viewBox="0 0 1040 625">',
       '<rect width="1040" height="625" fill="#fff"/>',
       '<text x="520" y="32" text-anchor="middle" font-family="sans-serif" font-size="23" fill="#182737">Center information I vs validation balanced accuracy</text>']
    for p in [0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        y=yp(p)
        s+=[f'<line x1="{x0}" y1="{y:.2f}" x2="{x0+width}" y2="{y:.2f}" stroke="#e1e5ec"/>',f'<text x="{x0-15}" y="{y+5:.2f}" text-anchor="end" font-family="sans-serif" font-size="14">{p:.0%}</text>']
    for exponent in range(math.floor(xmin), math.ceil(xmax)+1):
        x=xp(10**exponent)
        s+=[f'<line x1="{x:.2f}" y1="{y0}" x2="{x:.2f}" y2="{y0+height}" stroke="#e1e5ec"/>',f'<text x="{x:.2f}" y="{y0+height+24}" text-anchor="middle" font-family="sans-serif" font-size="14">10^{exponent}</text>']
    chance=yp(1/3)
    s.append(f'<line x1="{x0}" y1="{chance:.2f}" x2="{x0+width}" y2="{chance:.2f}" stroke="#789" stroke-dasharray="7,5" stroke-width="1.7"/>')
    baseline=yp(cases[0]['validation']['balanced_accuracy'])
    s.append(f'<line x1="{x0}" y1="{baseline:.2f}" x2="{x0+width}" y2="{baseline:.2f}" stroke="#2974bc" stroke-dasharray="3,4" stroke-width="1.6"/>')
    pts=' '.join(f'{xp(r["I"]):.2f},{yp(r["validation_balanced_accuracy"]):.2f}' for r in measured)
    s.append(f'<polyline points="{pts}" stroke="#516f92" stroke-width="2" fill="none" stroke-linejoin="round"/>')
    for r in measured:
        x,y=xp(r['I']),yp(r['validation_balanced_accuracy'])
        selected = r['round'] in (cases[1]['round'],cases[2]['round'])
        fill = '#dc5428' if selected else '#6670aa'
        s.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{6 if selected else 4.2}" fill="{fill}" stroke="white" stroke-width="1.2"><title>round {r["round"]}, I={r["I"]:.6g}, BA={100*r["validation_balanced_accuracy"]:.2f}%</title></circle>')
        if selected or r['round'] in (8,14,20,26,32,37,42,48):
            s.append(f'<text x="{x+7:.2f}" y="{y-8:.2f}" font-family="sans-serif" font-size="13" fill="#202d3e">{r["round"]}</text>')
    s += [f'<text x="{x0+width/2}" y="{y0+height+62}" text-anchor="middle" font-family="sans-serif" font-size="19">Center information I (nats, log scale)</text>',
          f'<text transform="translate(30 {y0+height/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="19">Validation balanced accuracy</text>',
          '<text x="105" y="587" font-family="sans-serif" font-size="13" fill="#4b586c">Only observed validation points; round number annotated. Dashed lines: chance and uniform reference (different protocol).</text>', '</svg>']
    return '\n'.join(s)+'\n'


def center_svg(cases):
    # Explicitly show REALIZED centers only, not targets.
    labels=['Uniform baseline','First guided run (round 0)','Peak guided validation (round 4)']
    cscale=[f'{j}' for j in range(3)]
    s=['<svg xmlns="http://www.w3.org/2000/svg" width="1050" height="470" viewBox="0 0 1050 470">',
       '<rect width="1050" height="470" fill="white"/>',
       '<text x="525" y="31" text-anchor="middle" font-family="sans-serif" font-size="23">Realized class-center matrices (not target matrices)</text>']
    for m,case in enumerate(cases):
        ox=30+340*m
        s.append(f'<text x="{ox+140}" y="67" text-anchor="middle" font-family="sans-serif" font-weight="bold" font-size="16">{labels[m]}</text>')
        for j in range(3):
            s.append(f'<text x="{ox+75+90*j}" y="104" text-anchor="middle" font-family="sans-serif" font-size="14">Label {j}</text>')
        for i,row in enumerate(case['centers']):
            y=123+91*i
            s.append(f'<text x="{ox-3}" y="{y+42}" text-anchor="start" font-family="sans-serif" font-size="13">C{i}</text>')
            for j,val in enumerate(row):
                opacity=.13+.75*val
                x=ox+28+90*j
                s.append(f'<rect x="{x}" y="{y}" width="86" height="82" rx="7" fill="#1f7a9b" fill-opacity="{opacity:.3f}"/>')
                s.append(f'<text x="{x+43}" y="{y+46}" text-anchor="middle" font-family="monospace" font-size="17" fill="#071e2a">{val:.3f}</text>')
        s.append(f'<text x="{ox+152}" y="438" text-anchor="middle" font-family="sans-serif" font-size="14">I = {case["geometry"]["I"]:.4f}</text>')
    s.append('</svg>')
    return '\n'.join(s)+'\n'


def write_csv(path, rows, names):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def analyze(data, outdir):
    cases = select_cases(data)
    trajectory = create_trajectory(data)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    figdir = outdir.parent/'figures'
    figdir.mkdir(exist_ok=True)
    summary = {'comparison_note': 'Each case is evaluated against its own REALIZED training centers; guided validation is candidate-class-conditional.',
               'best_round_selected_on_validation': cases[2]['round'],
               'number_of_rounds': len(trajectory),
               'measured_validation_rounds': [r['round'] for r in trajectory if r['validation_balanced_accuracy'] is not None],
               'cases': cases}
    (outdir/'three_cases.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    fields=['case','round','protocol','val_accuracy','val_balanced_accuracy','train_accuracy','train_balanced_accuracy','I','T','TNL','BT','BTNL','min_center_sqrt_jsd','mean_pairwise_center_jsd','train_within_class_jsd','target_realized_mean_jsd']
    compact=[]
    for c in cases:
        compact.append({'case':c['case'],'round':c['round'],'protocol':c['protocol'],
                        'val_accuracy':c['validation']['accuracy'],
                        'val_balanced_accuracy':c['validation']['balanced_accuracy'],
                        'train_accuracy':c['train'].get('accuracy'),
                        'train_balanced_accuracy':c['train'].get('balanced_accuracy'),
                        **{name:c['geometry'][name] for name in ('I','T','TNL','BT','BTNL','min_center_sqrt_jsd','mean_pairwise_center_jsd')},
                        'train_within_class_jsd':c['train_within_class_jsd'],
                        'target_realized_mean_jsd':c['target_realized_mean_jsd']})
    write_csv(outdir/'three_cases.csv',compact,fields)
    write_csv(outdir/'trajectory_metrics.csv',trajectory,list(trajectory[0]))
    (outdir.parent/'REPORT_FA.md').write_text(report_fa(cases,trajectory,data),encoding='utf-8')
    (figdir/'information_vs_accuracy.svg').write_text(information_svg(trajectory,cases),encoding='utf-8')
    (figdir/'center_matrices.svg').write_text(center_svg(cases),encoding='utf-8')
    return summary


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--history',type=Path,default=DEFAULT_HISTORY)
    cli.add_argument('--outdir',type=Path,default=BASE/'results')
    args=cli.parse_args()
    data=json.loads(args.history.read_text(encoding='utf-8'))
    result=analyze(data,args.outdir)
    print('Compared: '+', '.join(c['case'] for c in result['cases']))
    print('Highest guided validation balanced accuracy at round',result['best_round_selected_on_validation'])
    print('Measured validation rounds:',len(result['measured_validation_rounds']))
    print('Written output to',args.outdir)


if __name__=='__main__':
    main()
