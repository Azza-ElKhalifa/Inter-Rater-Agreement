# Figure 3: sensing-modalities heatmap (33 included studies).
# Requires: studies_33_extraction.csv (same folder).
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import ListedColormap

PASTEL_GREEN = '#A8D5BA'
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans']})

papers = list(csv.DictReader(open('studies_33_extraction.csv', encoding='utf-8-sig')))
print(f'Studies plotted: {len(papers)}')

def author_label(p):
    authors = str(p.get('Authors', ''))
    first = authors.split(' et al')[0].split(',')[0].split(' and ')[0].strip()
    year = str(p.get('Year', '')).replace('/', '\n')
    return f'{first} ({year})'

modality_map = {
    'EEG': ['eeg', 'single-lead eeg', 'behind-the-ear eeg', 'in-ear eeg'],
    'ECG/HR': ['ecg', 'hr', 'heart rate', 'hrv', 'ecg (primary)', 'ecg (hrv)', 'hr (bvp)'],
    'EDA/GSR': ['eda', 'gsr', 'gsr/eda'],
    'BVP/PPG': ['bvp', 'ppg', 'bvp/ppg', 'bvp (ppg)', 'in-ear ppg', 'rppg', 'rppg (from video)'],
    'EMG': ['emg', 'emg (x3)'],
    'Respiratory': ['resp', 'rsp', 'respiratory', 'respiration', 'respiratory belt'],
    'Temperature': ['temp', 'tmp', 'skin temp', 'fingertip temp', 'body temp', 'skt'],
    'ACC/Gyro': ['acc', 'gyro', 'accelerometer', 'gyroscope'],
    'Eye': ['eye movement', 'eye tracking', 'eog'],
    'Facial/Video': ['facial expressions (video)', 'facial (periocular)', 'facial complexion (kinect)',
                     'facial video', 'visual (video)'],
    'Speech': ['speech', 'speech (mfcc)'],
    'fNIRS': ['fnirs'],
    'Other': ['sbp', 'dbp', 'tpr', 'air pressure', 'uv', 'noise', 'gps',
              'keystroke', 'ibi', '8 physiological'],
}
modality_names = list(modality_map.keys())

matrix = np.zeros((len(modality_names), len(papers)), dtype=int)
for j, p in enumerate(papers):
    mods_lower = str(p.get('Sensing Modalities', '')).lower()
    for i, (mod_name, keywords) in enumerate(modality_map.items()):
        for kw in keywords:
            if kw in mods_lower:
                matrix[i, j] = 1
                break

for name, row in zip(modality_names, matrix):
    print(f'  {name}: {row.sum()}')

labels = [author_label(p) for p in papers]

fig, ax = plt.subplots(figsize=(26, 11))
cmap = ListedColormap(['#F5F5F5', PASTEL_GREEN])
ax.imshow(matrix, cmap=cmap, aspect='auto', interpolation='nearest')

ax.set_xticks(np.arange(len(papers)))
ax.set_yticks(np.arange(len(modality_names)))
ax.set_xticklabels(labels, rotation=75, ha='right', fontsize=26)
ax.set_yticklabels(modality_names, fontsize=28)

ax.set_xticks(np.arange(-0.5, len(papers), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(modality_names), 1), minor=True)
ax.grid(which='minor', color='white', linewidth=1.5)
ax.tick_params(which='minor', size=0)

present_patch = mpatches.Patch(facecolor=PASTEL_GREEN, edgecolor='gray', label='Used')
absent_patch = mpatches.Patch(facecolor='#F5F5F5', edgecolor='gray', label='Not used')
ax.legend(handles=[present_patch, absent_patch], loc='center left',
          bbox_to_anchor=(1.01, 0.5), ncol=1, fontsize=27, frameon=True,
          handlelength=1.9, handleheight=1.15, borderpad=0.65, labelspacing=0.7)

plt.tight_layout()
plt.savefig('Fig3.png', dpi=500, bbox_inches='tight')
plt.close()
print('Saved Fig3.png')
