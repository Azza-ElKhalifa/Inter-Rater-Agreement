# Recomputes every descriptive count reported in the manuscript from the
# data files (33 included studies).
import csv
from collections import Counter

papers = list(csv.DictReader(open('studies_extraction_33.csv', encoding='utf-8-sig')))
lims = list(csv.DictReader(open('dataset_dependence_analysis/limitations_33.csv', encoding='utf-8-sig')))
rtfp = list(csv.DictReader(open('dataset_dependence_analysis/rtfp_fourlevel_33.csv', encoding='utf-8-sig')))
assert len(papers) == len(lims) == len(rtfp) == 33

print('Included studies:', len(papers))

years = Counter(int(str(p['Year']).split('/')[-1]) for p in papers)
print('\nPublication years:', dict(sorted(years.items())))

modality_map = {
    'EEG': ['eeg'], 'ECG/HR': ['ecg', 'hr', 'heart rate', 'hrv'], 'EDA/GSR': ['eda', 'gsr'],
    'BVP/PPG': ['bvp', 'ppg', 'rppg'], 'EMG': ['emg'],
    'Respiratory': ['resp', 'rsp', 'respiration', 'respiratory belt'],
    'Temperature': ['temp', 'tmp', 'skt'], 'ACC/Gyro': ['acc', 'gyro'],
    'Eye': ['eye movement', 'eye tracking', 'eog'],
    'Facial/Video': ['facial', 'visual (video)'], 'Speech': ['speech'], 'fNIRS': ['fnirs'],
}
print('\nModality counts:')
for m, kws in modality_map.items():
    n = sum(1 for p in papers if any(k in str(p['Sensing Modalities']).lower() for k in kws))
    print(f'  {m}: {n}')

public_names = ['DEAP', 'WESAD', 'AMIGOS', 'MAHNOB-HCI', 'DREAMER', 'DECAF', 'SEED-IV', 'SEED',
                'SEED-FRA', 'ASCERTAIN', 'K-EmoCon', 'CASE', 'UBFC-Phys', 'POPANE', 'EMOGNITION',
                'MSSFT', 'HR-EEG4EMO', 'TERS-ER', 'Ulm-TSST', 'CVDiMo']
ds = Counter()
selfc = 0
for p in papers:
    d = str(p['Dataset(s) Used']).lower()
    if any(x in d for x in ['self-collected', 'self-col', 'custom']):
        selfc += 1
    for pub in public_names:
        if pub.lower() in d:
            ds[pub] += 1
print('\nSelf-collected:', selfc)
print('Public dataset use:', dict(ds.most_common()))

emo = Counter()
for p in papers:
    e = str(p['Emotion Representation']).lower()
    hd, hdim, hb = 'discrete' in e, 'dimensional' in e or '4-quadrant' in e, e.startswith('binary')
    emo['Both' if (hd and (hdim or hb)) else 'Discrete' if hd else 'Dimensional' if hdim
        else 'Binary' if hb else 'Other'] += 1
print('\nEmotion representation:', dict(emo))

fus = Counter()
for p in papers:
    for part in str(p['Fusion Strategy']).split(';'):
        part = part.strip()
        if part and part not in ('Not reported', 'Not specified', 'Not detailed', 'Not mentioned', 'N/A'):
            fus[part] += 1
print('\nFusion strategies (>=2):', {k: v for k, v in fus.most_common() if v >= 2})

print('\nLimitation profile (high severity, mean):')
for L in ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']:
    v = [int(r[L]) for r in lims]
    hi = sum(1 for x in v if x == 3)
    print(f'  {L}: high {hi}/33 ({hi/33*100:.1f}%), mean {sum(v)/33:.2f}')

domains = [('Technical validation', [3, 4]), ('Naturalistic/longitudinal', [2, 5]),
           ('Human factors/usability', [1, 6]), ('Application utility', [9]),
           ('Safety/privacy/security', [7, 8]), ('Regulation/implementation', [10])]
print('\nRTFP domain profile (not reported / discussed / implemented / evaluated):')
for name, cs in domains:
    v = [int(r[f'C{c}']) for r in rtfp for c in cs]
    n = len(v)
    pct = [sum(1 for x in v if x == k) / n * 100 for k in range(4)]
    print(f'  {name}: ' + ' / '.join(f'{p:.1f}%' for p in pct))
