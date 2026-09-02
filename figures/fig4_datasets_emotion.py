# Figure 4: (a) dataset usage, (b) emotion representation (33 included studies).
# Requires: studies_33_extraction.csv (same folder).
import csv
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                     'savefig.facecolor': 'white', 'savefig.dpi': 500, 'savefig.bbox': 'tight'})

PASTEL_GREEN  = '#A8D5BA'
PASTEL_YELLOW = '#F5DEB3'
PASTEL_PINK   = '#D4A5C4'
PASTEL_BLUE   = '#A8C8E8'
PASTEL_PEACH  = '#F5C6AA'

papers = list(csv.DictReader(open('studies_33_extraction.csv', encoding='utf-8-sig')))
print(f'Loaded {len(papers)} papers')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7),
                               gridspec_kw={'width_ratios': [1.2, 1]})

# Panel (a): dataset usage (lollipop)
public_datasets = Counter()
custom_count = 0

public_names = ['DEAP', 'WESAD', 'AMIGOS', 'MAHNOB-HCI', 'DREAMER', 'DECAF',
                'SEED-IV', 'SEED', 'SEED-FRA', 'ASCERTAIN', 'K-EmoCon',
                'CASE', 'UBFC-Phys', 'POPANE', 'EMOGNITION', 'MSSFT',
                'HR-EEG4EMO', 'BAUM-1s', 'RML', 'eNTERFACE05',
                'TERS-ER', 'Ulm-TSST', 'CVDiMo']

for p in papers:
    ds_lower = str(p.get('Dataset(s) Used', '')).lower()
    if any(x in ds_lower for x in ['self-collected', 'self-col', 'custom']):
        custom_count += 1
    for pub in public_names:
        if pub.lower() in ds_lower:
            public_datasets[pub] += 1

dataset_counts = {}
for name, count in public_datasets.most_common():
    if count >= 2:
        dataset_counts[name] = count
    else:
        dataset_counts['Other public'] = dataset_counts.get('Other public', 0) + 1
dataset_counts['Self-collected'] = custom_count

sorted_items = sorted(dataset_counts.items(), key=lambda x: x[1], reverse=True)
ds_names = [x[0] for x in sorted_items]
ds_counts = [x[1] for x in sorted_items]
print('Datasets:', dict(zip(ds_names, ds_counts)))

DOT_TOP = '#E07B7B'
DOT_MID = '#B5B867'
DOT_LOW = '#D4A5C4'

dot_colors = []
sep_idx = None
for i, (name, count) in enumerate(zip(ds_names, ds_counts)):
    if i < 2:
        dot_colors.append(DOT_TOP)
    elif count >= 3:
        dot_colors.append(DOT_MID)
    else:
        dot_colors.append(DOT_LOW)
        if sep_idx is None:
            sep_idx = i

y_pos = range(len(ds_names))
for i, count in enumerate(ds_counts):
    ax1.plot([0, count], [i, i], color='#CCCCCC', linewidth=1, zorder=1)
ax1.scatter(ds_counts, y_pos, c=dot_colors, s=120, zorder=2, edgecolors='white', linewidth=1.5)
for i, count in enumerate(ds_counts):
    ax1.text(count + 0.4, i, str(count), ha='left', va='center',
             fontsize=17, fontweight='bold')

ax1.set_yticks(y_pos)
ax1.set_yticklabels(ds_names, fontsize=18)
ax1.invert_yaxis()
ax1.set_xlabel('Number of Studies', fontsize=17)
ax1.set_xlim(0, max(ds_counts) + 3)
ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
if sep_idx is not None:
    ax1.axhline(y=sep_idx - 0.5, color='#CCCCCC', linestyle='--', linewidth=0.8)

ax1.text(0, 1.03, r'$\bf{a}$', fontsize=18, transform=ax1.transAxes, ha='left', va='bottom')

# Panel (b): emotion representation (donut)
categories = Counter()
for p in papers:
    e_lower = str(p.get('Emotion Representation', '')).lower()
    has_discrete = 'discrete' in e_lower
    has_dimensional = 'dimensional' in e_lower or '4-quadrant' in e_lower
    has_binary = e_lower.startswith('binary')

    if has_discrete and has_dimensional:
        categories['Both'] += 1
    elif has_discrete and has_binary:
        categories['Both'] += 1
    elif has_discrete:
        categories['Discrete'] += 1
    elif has_dimensional:
        categories['Dimensional'] += 1
    elif has_binary:
        categories['Binary'] += 1
    else:
        categories['Other'] += 1
print('Emotion representation:', dict(categories))

em_labels = list(categories.keys())
em_sizes = list(categories.values())
em_colors = [PASTEL_GREEN, PASTEL_YELLOW, PASTEL_PINK, PASTEL_BLUE, PASTEL_PEACH][:len(em_labels)]

wedges, texts, autotexts = ax2.pie(em_sizes, labels=em_labels, colors=em_colors,
                                   autopct=lambda pct: f'{int(round(pct/100.*sum(em_sizes)))} ({pct:.0f}%)',
                                   pctdistance=0.75, startangle=90,
                                   wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2))
for t in texts:
    t.set_fontsize(18)
for t in autotexts:
    t.set_fontsize(15)
    t.set_fontweight('bold')

ax2.text(0, 0, f'n={sum(em_sizes)}', ha='center', va='center',
         fontsize=19, fontweight='bold', color='#444444')

ax2.text(0, 1.03, r'$\bf{b}$', fontsize=18, transform=ax2.transAxes, ha='left', va='bottom')

plt.tight_layout()
plt.savefig('Fig4.png')
plt.close()
print('Saved Fig4.png')
