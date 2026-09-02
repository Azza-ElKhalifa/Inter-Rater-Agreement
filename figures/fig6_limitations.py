# Figure 6: (a) per-study limitation severity heatmap, (b) distribution per criterion (33 included studies).
# Requires: limitations_33.csv and studies_33_extraction.csv (same folder).
import csv
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import ListedColormap

HEATMAP_GREEN  = '#A8D5BA'
HEATMAP_YELLOW = '#F5DEB3'
HEATMAP_PINK   = '#D4A5C4'
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                     'savefig.facecolor': 'white', 'savefig.dpi': 500, 'savefig.bbox': 'tight'})

lim_rows = list(csv.DictReader(open('limitations_33.csv', encoding='utf-8-sig')))
papers = list(csv.DictReader(open('studies_33_extraction.csv', encoding='utf-8-sig')))
assert len(lim_rows) == len(papers)

lim_categories = OrderedDict([
    ('L1', 'Sample Size Adequacy'),
    ('L2', 'Population Diversity'),
    ('L3', 'Ecological Validity'),
    ('L4', 'Validation Strategy'),
    ('L5', 'Dataset Generalisability'),
    ('L6', 'Reference-Standard Validity'),
])

n_cats = len(lim_categories)
n_papers = len(lim_rows)
matrix = np.zeros((n_cats, n_papers), dtype=int)
for j, r in enumerate(lim_rows):
    for i, L in enumerate(['L1', 'L2', 'L3', 'L4', 'L5', 'L6']):
        matrix[i, j] = int(r[L]) - 1

def author_label(p):
    authors = str(p.get('Authors', ''))
    first = authors.split(' et al')[0].split(',')[0].split(' and ')[0].strip()
    year = str(p.get('Year', '')).replace('/', '\n')
    return f'{first} ({year})'

labels = [author_label(p) for p in papers]
cat_labels = list(lim_categories.keys())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 11),
                               gridspec_kw={'width_ratios': [2.5, 1]})

cmap = ListedColormap([HEATMAP_GREEN, HEATMAP_YELLOW, HEATMAP_PINK])
ax1.imshow(matrix, cmap=cmap, aspect='auto', interpolation='nearest', vmin=0, vmax=2)

ax1.set_xticks(np.arange(n_papers))
ax1.set_yticks(np.arange(n_cats))
ax1.set_xticklabels(labels, rotation=75, ha='right', fontsize=24)
ax1.set_yticklabels(cat_labels, fontsize=27)

ax1.set_xticks(np.arange(-0.5, n_papers, 1), minor=True)
ax1.set_yticks(np.arange(-0.5, n_cats, 1), minor=True)
ax1.grid(which='minor', color='white', linewidth=1.5)
ax1.tick_params(which='minor', size=0)

ax1.text(0, 1.02, r'$\bf{a}$', fontsize=24, transform=ax1.transAxes, ha='left', va='bottom')

# Panel (b): stacked horizontal bars
low_pct, mod_pct, high_pct = [], [], []
for i in range(n_cats):
    low_pct.append(np.sum(matrix[i] == 0) / n_papers * 100)
    mod_pct.append(np.sum(matrix[i] == 1) / n_papers * 100)
    high_pct.append(np.sum(matrix[i] == 2) / n_papers * 100)
for L, lo, mo, hi in zip(cat_labels, low_pct, mod_pct, high_pct):
    print(f'  {L}: low {lo:.1f}% | moderate {mo:.1f}% | high {hi:.1f}%')

y_pos = np.arange(n_cats)
ax2.barh(y_pos, low_pct, color=HEATMAP_GREEN, edgecolor='white', height=0.7, label='Low')
ax2.barh(y_pos, mod_pct, left=low_pct, color=HEATMAP_YELLOW, edgecolor='white', height=0.7, label='Moderate')
ax2.barh(y_pos, high_pct, left=[l+m for l, m in zip(low_pct, mod_pct)],
         color=HEATMAP_PINK, edgecolor='white', height=0.7, label='High')

ax2.set_yticks(y_pos)
ax2.set_yticklabels(cat_labels, fontsize=27)
ax2.invert_yaxis()
ax2.set_xlim(0, 100)
ax2.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
ax2.set_xticklabels(['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'],
                    fontsize=19)

ax2.text(0, 1.02, r'$\bf{b}$', fontsize=24, transform=ax2.transAxes, ha='left', va='bottom')

low_patch = mpatches.Patch(facecolor=HEATMAP_GREEN, edgecolor='gray', label='Low')
mod_patch = mpatches.Patch(facecolor=HEATMAP_YELLOW, edgecolor='gray', label='Moderate')
high_patch = mpatches.Patch(facecolor=HEATMAP_PINK, edgecolor='gray', label='High')

fig.legend(handles=[low_patch, mod_patch, high_patch],
           title='Limitation levels:', title_fontsize=24,
           loc='lower right', bbox_to_anchor=(1.0, 0.02),
           fontsize=22, frameon=True, fancybox=True)

lim_text = '\n'.join([f'{code}:   {desc}' for code, desc in lim_categories.items()])
fig.text(0.72, 0.02, 'Limitations:\n' + lim_text,
         fontsize=23, va='bottom', ha='left', family='sans-serif')

plt.tight_layout()
plt.subplots_adjust(bottom=0.38, wspace=0.15)
plt.savefig('Fig6.png')
plt.close()
print('Saved Fig6.png')
