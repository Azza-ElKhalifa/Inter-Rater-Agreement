# Dataset-dependence sensitivity analysis (33 included articles).
#
# Several articles analyze the same public dataset, so the 33 articles are not
# 33 independent cohorts. Rule, stated before the analysis was run:
#   Primary dataset = the dataset on which the article's best reported result
#   was obtained; when that result was not tied to a single dataset, the
#   first-listed dataset was used. For every public dataset used as primary by
#   two or more articles, one representative article was retained: the most
#   recently published, with ties resolved by the larger total analyzed sample.
#   All self-collected cohorts were retained as genuinely independent.
#
# Outputs: full-set vs reduced-set criterion-level results, plus a
# leave-one-dataset-out analysis, written to sensitivity_results.txt.
import csv
from collections import defaultdict

assign = list(csv.DictReader(open('dataset_cluster_assignments.csv', encoding='utf-8-sig')))
lims = list(csv.DictReader(open('limitations_33.csv', encoding='utf-8-sig')))
rtfp = list(csv.DictReader(open('rtfp_fourlevel_33.csv', encoding='utf-8-sig')))
assert len(assign) == len(lims) == len(rtfp) == 33

def lim_profile(idx):
    n = len(idx)
    out = {}
    for L in ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']:
        hi = sum(1 for i in idx if int(lims[i][L]) == 3)
        out[L] = hi / n * 100
    return out

def rtfp_profile(idx):
    n = len(idx)
    def vals(cs):
        return [int(rtfp[i][f'C{c}']) for i in idx for c in cs]
    v34 = vals([3, 4]); v25 = vals([2, 5]); v78 = vals([7, 8]); v16 = vals([1, 6])
    v9 = vals([9]); v10 = vals([10])
    return {
        'C3-C4 not reported': sum(1 for v in v34 if v == 0) / len(v34) * 100,
        'C2,C5 not reported': sum(1 for v in v25 if v == 0) / len(v25) * 100,
        'C7-C8 not reported': sum(1 for v in v78 if v == 0) / len(v78) * 100,
        'C1,C6 implemented': sum(1 for v in v16 if v == 2) / len(v16) * 100,
        'C9 discussed': sum(1 for v in v9 if v == 1) / len(v9) * 100,
        'C10 not reported': sum(1 for v in v10 if v == 0) / len(v10) * 100,
    }

def report(name, idx, lines):
    lp = lim_profile(idx)
    rp = rtfp_profile(idx)
    lines.append(f'{name} (n = {len(idx)})')
    lines.append('  High-severity limitations: ' + '; '.join(f'{k} {v:.1f}%' for k, v in lp.items()))
    for k, v in rp.items():
        lines.append(f'  {k}: {v:.1f}%')
    lines.append('')
    return lp, rp

full_idx = list(range(33))
reduced_idx = [i for i, a in enumerate(assign) if a['Retained in reduced set'] == 'Yes']

lines = ['Dataset-dependence sensitivity analysis', '=' * 40, '']
full_lp, full_rp = report('Full set', full_idx, lines)
red_lp, red_rp = report('Reduced set (one representative per repeated public dataset)', reduced_idx, lines)

lines.append('Absolute change, reduced vs full (percentage points):')
for k in full_lp:
    lines.append(f'  {k} high: {red_lp[k] - full_lp[k]:+.1f}')
for k in full_rp:
    lines.append(f'  {k}: {red_rp[k] - full_rp[k]:+.1f}')
lines.append('')

# Leave-one-dataset-out: drop all articles whose primary dataset is D, for each public D
lines.append('Leave-one-dataset-out (all articles with that primary dataset removed):')
clusters = defaultdict(list)
for i, a in enumerate(assign):
    if a['Cohort type'] == 'Public':
        clusters[a['Primary dataset or cohort']].append(i)
for ds in sorted(clusters, key=lambda d: -len(clusters[d])):
    idx = [i for i in full_idx if i not in clusters[ds]]
    lp = lim_profile(idx)
    rp = rtfp_profile(idx)
    lines.append(f'  Without {ds} (n = {len(idx)}): '
                 f'L3 high {lp["L3"]:.1f}%; C3-C4 NR {rp["C3-C4 not reported"]:.1f}%; '
                 f'C2,C5 NR {rp["C2,C5 not reported"]:.1f}%; C10 NR {rp["C10 not reported"]:.1f}%')
lines.append('')

mx_l = max(abs(red_lp[k] - full_lp[k]) for k in full_lp)
mx_r = max(abs(red_rp[k] - full_rp[k]) for k in full_rp)
lines.append(f'Largest absolute change in the reduced set: {max(mx_l, mx_r):.1f} percentage points.')

open('sensitivity_results.txt', 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
