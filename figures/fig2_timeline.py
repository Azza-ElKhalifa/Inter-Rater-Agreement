# Figure 2: publication timeline (33 included studies).
# Requires: studies_33_extraction.csv (same folder).
import csv
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                     'savefig.facecolor': 'white', 'savefig.dpi': 500, 'savefig.bbox': 'tight'})

papers = list(csv.DictReader(open('studies_33_extraction.csv', encoding='utf-8-sig')))
years = [int(str(p['Year']).split('/')[-1]) for p in papers]

year_counts = Counter(years)
all_years = list(range(2016, max(years) + 1))
counts = [year_counts.get(y, 0) for y in all_years]

cumulative = []
running = 0
for c in counts:
    running += c
    cumulative.append(running)

print('Years :', all_years)
print('Counts:', counts, '-> total', sum(counts))
print('Cumul :', cumulative)

fig, ax1 = plt.subplots(figsize=(8, 4.5))
bars = ax1.bar(all_years, counts, color='#D3D3D3', edgecolor='white', width=0.7, zorder=2)
for bar, count in zip(bars, counts):
    if count > 0:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                 str(count), ha='center', va='bottom', fontsize=10, fontweight='normal')

ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Number of Papers', fontsize=12)
ax1.set_xticks(all_years)
ax1.set_xticklabels([str(y) for y in all_years])
ax1.set_ylim(0, max(counts) + 2)
ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

ax2 = ax1.twinx()
ax2.plot(all_years, cumulative, color='black', marker='o', markersize=5,
         linewidth=1.5, zorder=3)
for x, y in zip(all_years, cumulative):
    ax2.text(x, y + 0.5, str(y), ha='center', va='bottom', fontsize=9, fontweight='normal')

ax2.set_ylabel('Cumulative Total', fontsize=12)
ax2.set_ylim(0, max(cumulative) + 3)
ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

plt.tight_layout()
plt.savefig('Fig2.png')
plt.close()
print('Saved Fig2.png')
