# Figure 7: (a) four-level RTFP domain profile, (b) main reporting gaps,
# (c) seven recommendations for future studies (33 included studies).
# Requires: rtfp_fourlevel_33.csv (same folder).
import csv
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

rows = list(csv.DictReader(open('rtfp_fourlevel_33.csv', encoding='utf-8-sig')))
N = len(rows)
assert N == 33

def col(c):
    return [int(r[f'C{c}']) for r in rows]

domains = [
    ('Technical validation', 'C3–C4', [3, 4]),
    ('Naturalistic/longitudinal', 'C2, C5', [2, 5]),
    ('Human factors/usability', 'C1, C6', [1, 6]),
    ('Application utility', 'C9', [9]),
    ('Safety/privacy/security', 'C7–C8', [7, 8]),
    ('Regulation/implementation', 'C10', [10]),
]
level_names = ['Not reported', 'Discussed', 'Implemented', 'Empirically evaluated']
LEVEL_COLORS = ['#D9D9D9', '#F5DEB3', '#A9C4DA', '#3C7D5E']

profile = []
for name, items, cs in domains:
    vals = [v for c in cs for v in col(c)]
    n = len(vals)
    pct = [sum(1 for v in vals if v == k) / n * 100 for k in range(4)]
    profile.append(pct)
    print(f'  {name} ({items}): ' + ' / '.join(f'{p:.1f}%' for p in pct))

DARK = '#22272E'
GREY = '#555555'
BOX_FACE, BOX_EDGE, BOX_ACCENT = '#F5EFE1', '#E3D8C2', '#C8912A'
GREEN_FACE, GREEN_EDGE, NUM = '#DCEEE4', '#3C7D5E', '#3C7D5E'

plt.rcParams.update({'font.family': 'Arial'})

fig = plt.figure(figsize=(12, 12.6))
axa = fig.add_axes([0.30, 0.695, 0.66, 0.235])
axb = fig.add_axes([0.03, 0.375, 0.94, 0.245])
axc = fig.add_axes([0.03, 0.035, 0.94, 0.255])

# ======================= PANEL (a) =======================
ypos = list(range(len(domains)))[::-1]
for (name, items, cs), pcts, y in zip(domains, profile, ypos):
    left = 0.0
    for k, (p, colr) in enumerate(zip(pcts, LEVEL_COLORS)):
        if p > 0:
            axa.barh(y, p, left=left, height=0.62, color=colr, zorder=3,
                     edgecolor='white', linewidth=0.8)
            if p >= 8:
                lbl = f'{p:.0f}%' if abs(p - round(p)) < 1e-9 else f'{p:.1f}%'
                axa.text(left + p / 2, y, lbl, va='center', ha='center',
                         fontsize=11.5, fontweight='bold',
                         color='white' if k == 3 else DARK)
        left += p
    axa.text(-2, y, name, va='center', ha='right', fontsize=13.5, color=DARK)

axa.set_xlim(0, 100)
axa.set_ylim(-0.6, len(domains) - 0.4)
axa.set_xticks([0, 25, 50, 75, 100])
axa.set_xticklabels(['0', '25', '50', '75', '100'], fontsize=12, color=DARK)
axa.set_yticks([])
axa.set_xlabel('Criterion-level assessments (%)', fontsize=13, color=GREY)
for sp in ['top', 'right', 'left', 'bottom']:
    axa.spines[sp].set_visible(False)
axa.tick_params(axis='both', length=0)
handles = [mpatches.Patch(facecolor=c, edgecolor='#BBBBBB', label=l)
           for c, l in zip(LEVEL_COLORS, level_names)]
axa.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.02),
           ncol=4, fontsize=11.5, frameon=False, columnspacing=1.2, handlelength=1.4)
fig.text(0.03, 0.965, 'a', fontsize=21, fontweight='bold', va='top', color=DARK)

# ======================= PANEL (b) =======================
c6 = Counter(col(6)); c7 = Counter(col(7)); c8 = Counter(col(8)); c9 = Counter(col(9)); c10 = Counter(col(10))
gaps = [
    ('Explainability', f'Not reported in {c6[0]} of {N} studies'),
    ('Missing-data handling', f'Not reported in {c7[0]} of {N} studies'),
    ('Privacy and security', f'Not reported in {c8[0]} of {N} studies'),
    ('Application-specific utility', f'Discussed in {c9[1]} of {N} studies but not implemented or evaluated'),
    ('Regulatory considerations', f'Not reported in any of the {N} studies'),
]
axb.set_xlim(0, 100); axb.set_ylim(0, 100); axb.axis('off')
fig.text(0.03, 0.645, 'b', fontsize=21, fontweight='bold', va='top', color=DARK)
x0, x1 = 0.8, 99.2
bw = x1 - x0
top, bh, gap = 84, 12.5, 3.2
for i, (title, q) in enumerate(gaps):
    y = top - i * (bh + gap)
    box = FancyBboxPatch((x0, y - bh), bw, bh, boxstyle='round,pad=0.4,rounding_size=2.5',
                         facecolor=BOX_FACE, edgecolor=BOX_EDGE, linewidth=1.2)
    axb.add_patch(box)
    axb.add_patch(plt.Rectangle((x0 + 1.2, y - bh + 1.2), 1.4, bh - 2.4,
                                facecolor=BOX_ACCENT, edgecolor='none'))
    axb.text(x0 + 4.5, y - bh / 2, title, va='center', ha='left', fontsize=15.5,
             fontweight='bold', color=DARK)
    axb.text(x1 - 2.5, y - bh / 2, q, va='center', ha='right', fontsize=14, color=GREY)

# ======================= PANEL (c) =======================
axc.set_xlim(0, 100); axc.set_ylim(0, 100); axc.axis('off')
fig.text(0.03, 0.305, 'c', fontsize=21, fontweight='bold', va='top', color=DARK)
steps = [
    'Naturalistic and\nlongitudinal evaluation',
    'Validation appropriate\nto the intended use',
    'Justified sample sizes and\nrepresentative populations',
    'Plan explainability, privacy and\nregulation from the start',
    'Report translational features\nalongside accuracy',
    'Wearability-oriented\nsensor selection',
    'Transparent validation-\npipeline reporting',
]
yc = 50
xs = [8 + i * 14 for i in range(7)]
axc.annotate('', xy=(xs[-1] + 6, yc), xytext=(xs[0] - 4, yc),
             arrowprops=dict(arrowstyle='-|>', color=GREEN_EDGE, linewidth=2.5))
for i, (x, label) in enumerate(zip(xs, steps)):
    axc.plot(x, yc, marker='o', markersize=34, markerfacecolor=GREEN_FACE,
             markeredgecolor=GREEN_EDGE, markeredgewidth=2.4, zorder=3)
    axc.text(x, yc, str(i + 1), va='center', ha='center', fontsize=15,
             fontweight='bold', color=NUM, zorder=4)
    above = (i % 2 == 0)
    axc.text(x, yc + (17 if above else -17), label, va='center', ha='center',
             fontsize=12, color=DARK)

fig.savefig('Fig7.png', dpi=500, facecolor='white')
plt.close()
print('Saved Fig7.png')
