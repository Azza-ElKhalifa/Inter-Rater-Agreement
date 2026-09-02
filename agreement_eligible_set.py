# Inter-rater agreement on the final eligible set (33 articles):
# four-level RTFP (330 ratings) and L1-L6 (198 ratings),
# Cohen's kappa with 95% bootstrap CI (2000 iterations, seed 42).
import csv
import numpy as np
from sklearn.metrics import cohen_kappa_score


def kappa_bootstrap_ci(y1, y2, weights=None, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    y1, y2 = np.array(y1), np.array(y2)
    n = len(y1)
    kappa = cohen_kappa_score(y1, y2, weights=weights)
    boot = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        k = cohen_kappa_score(y1[idx], y2[idx], weights=weights)
        boot.append(1.0 if np.isnan(k) else k)
    return kappa, float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))


four = [r for r in csv.DictReader(open('rtfs_fourlevel_comparison.csv', encoding='utf-8-sig'))
        if not r['Study'].lower().startswith(('qin', 'wu'))]
lims = [r for r in csv.DictReader(open('limitation_scores_comparison.csv', encoding='utf-8-sig'))
        if not r['Study'].lower().startswith(('qin', 'wu'))]
assert len(four) == len(lims) == 33

ae4 = [int(r[f'AE_C{c}']) for r in four for c in range(1, 11)]
mk4 = [int(r[f'MK_C{c}']) for r in four for c in range(1, 11)]
names = ['SampleSize', 'Diversity', 'Ecological', 'Validation', 'Dataset', 'RefStandard']
aeL = [int(r[f'AE_L{i}_{n}']) for r in lims for i, n in enumerate(names, 1)]
mkL = [int(r[f'MK_L{i}_{n}']) for r in lims for i, n in enumerate(names, 1)]

lines = ['Inter-rater agreement, final eligible set (33 articles)', '=' * 55, '']
res = [
    ('Four-level RTFP (quadratic)', ae4, mk4, 'quadratic'),
    ('L1-L6 limitations (quadratic)', aeL, mkL, 'quadratic'),
    ('L1-L6 limitations (linear)', aeL, mkL, 'linear'),
]
for name, a, b, w in res:
    k, lo, hi = kappa_bootstrap_ci(a, b, weights=w)
    lines.append(f'{name}: n = {len(a)}; kappa = {k:.4f} [{lo:.4f} - {hi:.4f}]')
lines.append('')
lines.append('CI: 2000-iteration bootstrap of rating pairs, seed 42.')
open('agreement_eligible_set_results.txt', 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
