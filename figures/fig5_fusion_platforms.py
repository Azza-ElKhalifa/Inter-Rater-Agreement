# Figure 5: (a) fusion strategies, (b) sensor platforms (33 included studies).
# Requires: studies_33_extraction.csv (same folder).
import csv
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                     'savefig.facecolor': 'white', 'savefig.dpi': 500, 'savefig.bbox': 'tight'})

PASTEL_YELLOW   = '#F5DEB3'
PASTEL_LAVENDER = '#C8B8DB'

papers = list(csv.DictReader(open('studies_33_extraction.csv', encoding='utf-8-sig')))
print(f'Loaded {len(papers)} papers')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7),
                               gridspec_kw={'width_ratios': [1.2, 1]})

# Panel (a): fusion strategies (lollipop)
fusion_counts = Counter()
for p in papers:
    for part in str(p.get('Fusion Strategy', '')).split(';'):
        part = part.strip()
        if part and part not in ('Not reported', 'Not specified', 'Not detailed',
                                 'Not mentioned', 'N/A'):
            fusion_counts[part] += 1

sorted_items = sorted(fusion_counts.items(), key=lambda x: x[1], reverse=True)
sorted_items = [x for x in sorted_items if x[1] >= 2]
print('Fusion:', dict(sorted_items))

fu_names = [x[0] for x in sorted_items]
fu_counts = [x[1] for x in sorted_items]

y_pos = list(range(len(fu_names)))
for i, count in enumerate(fu_counts):
    ax1.plot([0, count], [i, i], color='#CCCCCC', linewidth=0.8,
             linestyle='--', zorder=1)

marker_colors = ['#555555' if i < 2 else '#AAAAAA' for i in range(len(fu_names))]
marker_sizes = [140 if i < 2 else 100 for i in range(len(fu_names))]

for i, (count, color, size) in enumerate(zip(fu_counts, marker_colors, marker_sizes)):
    ax1.scatter(count, i, marker='X', s=size, c=color, zorder=2)

for i, count in enumerate(fu_counts):
    ax1.text(count + 0.3, i, str(count), ha='left', va='center',
             fontsize=17, fontweight='bold')

ax1.set_yticks(y_pos)
ax1.set_yticklabels(fu_names, fontsize=19)
ax1.invert_yaxis()
ax1.set_xlabel('Number of Studies', fontsize=18)
ax1.set_xlim(0, max(fu_counts) + 2)
ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

ax1.text(0, 1.03, r'$\bf{a}$', fontsize=18, transform=ax1.transAxes, ha='left', va='bottom')

# Panel (b): sensor/platform type (pie)
platform_counts = Counter()
for p in papers:
    sensor = str(p.get('Sensor / Device Type', '')).lower()
    is_wearable = any(x in sensor for x in ['wearable', 'smartwatch', 'mobile', 'smartphone'])
    is_lab = 'lab' in sensor or 'vr' in sensor
    is_contactless = any(x in sensor for x in ['contactless', 'camera', 'kinect'])

    if is_wearable and is_lab:
        platform_counts['Hybrid'] += 1
    elif is_wearable and is_contactless:
        platform_counts['Hybrid'] += 1
    elif is_wearable:
        platform_counts['Wearable'] += 1
    elif is_contactless:
        platform_counts['Contactless'] += 1
    elif is_lab:
        platform_counts['Lab-based'] += 1
    else:
        platform_counts['Other'] += 1
print('Platforms:', dict(platform_counts))

sorted_plat = sorted(platform_counts.items(), key=lambda x: x[1], reverse=True)
pl_labels = [x[0] for x in sorted_plat]
pl_sizes = [x[1] for x in sorted_plat]

color_map = {
    'Wearable': '#A8D5E2',
    'Lab-based': '#F5C6C6',
    'Contactless': PASTEL_YELLOW,
    'Hybrid': PASTEL_LAVENDER,
    'Other': '#CCCCCC',
}
pl_colors = [color_map.get(l, '#CCCCCC') for l in pl_labels]
explode = [0.03 if s / sum(pl_sizes) < 0.15 else 0 for s in pl_sizes]

wedges, texts, autotexts = ax2.pie(
    pl_sizes, colors=pl_colors,
    autopct=lambda pct: f'{int(round(pct/100.*sum(pl_sizes)))} ({pct:.1f}%)',
    pctdistance=0.55, startangle=90,
    explode=explode,
    wedgeprops=dict(edgecolor='white', linewidth=2.5)
)
for t in autotexts:
    t.set_fontsize(16)

if len(autotexts) > 1:
    pos = autotexts[-1].get_position()
    autotexts[-1].set_position((pos[0] + 0.04, pos[1] + 0.20))
    autotexts[-1].set_fontsize(15)

legend_labels = [f'{l}' for l in pl_labels]
ax2.legend(wedges, legend_labels, loc='lower right', fontsize=17,
           frameon=False, bbox_to_anchor=(1.05, -0.15))

ax2.text(0.5, 1.0, f'n={sum(pl_sizes)}', fontsize=18, fontweight='bold',
         color='#444444', ha='center', va='bottom', transform=ax2.transAxes)

ax2.text(0, 1.03, r'$\bf{b}$', fontsize=18, transform=ax2.transAxes, ha='left', va='bottom')

plt.tight_layout()
plt.savefig('Fig5.png')
plt.close()
print('Saved Fig5.png')
