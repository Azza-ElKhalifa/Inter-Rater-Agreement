# Inter-rater agreement on the final eligible set (37 articles):
# four-level RTFP (370 ratings), L1-L6 (222 ratings), and the
# independent-reviewer L1-L6 agreement (222 ratings);
# Cohen's kappa with 95% bootstrap CI (2000 iterations, seed 42).
import csv
import numpy as np
from sklearn.metrics import cohen_kappa_score


def kappa_clustered_bootstrap_ci(y1, y2, group_size, weights=None, n_boot=2000, seed=42):
    # Pooled study-by-criterion ratings are ordered study-by-study in fixed-size
    # blocks (group_size criteria per study). Resampling whole studies (clusters)
    # preserves the within-study clustering of ratings; the point estimate is unchanged.
    rng = np.random.RandomState(seed)
    y1, y2 = np.array(y1), np.array(y2)
    n = len(y1)
    assert n % group_size == 0, (n, group_size)
    n_groups = n // group_size
    blocks = [np.arange(g * group_size, (g + 1) * group_size) for g in range(n_groups)]
    kappa = cohen_kappa_score(y1, y2, weights=weights)
    boot = []
    for _ in range(n_boot):
        chosen = rng.choice(n_groups, size=n_groups, replace=True)
        idx = np.concatenate([blocks[g] for g in chosen])
        k = cohen_kappa_score(y1[idx], y2[idx], weights=weights)
        boot.append(1.0 if np.isnan(k) else k)
    return kappa, float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))


four = [r for r in csv.DictReader(open('rtfs_fourlevel_comparison.csv', encoding='utf-8-sig'))
        if not r['Study'].lower().startswith(('qin', 'wu'))]
lims = [r for r in csv.DictReader(open('limitation_scores_comparison.csv', encoding='utf-8-sig'))
        if not r['Study'].lower().startswith(('qin', 'wu'))]
three = [r for r in csv.DictReader(open('limitation_scores_three_reviewers.csv', encoding='utf-8-sig'))
         if not r['Study'].lower().startswith(('qin', 'wu'))]
assert len(four) == len(lims) == len(three) == 37

ae4 = [int(r[f'AE_C{c}']) for r in four for c in range(1, 11)]
mk4 = [int(r[f'MK_C{c}']) for r in four for c in range(1, 11)]
names = ['SampleSize', 'Diversity', 'Ecological', 'Validation', 'Dataset', 'RefStandard']
aeL = [int(r[f'AE_L{i}_{n}']) for r in lims for i, n in enumerate(names, 1)]
mkL = [int(r[f'MK_L{i}_{n}']) for r in lims for i, n in enumerate(names, 1)]

lines = ['Inter-rater agreement, final eligible set (37 articles)', '=' * 55, '']
aeH = [int(r[f'AE_L{i}_{n}']) for r in three for i, n in enumerate(names, 1)]
hyL = [int(r[f'HY_L{i}_{n}']) for r in three for i, n in enumerate(names, 1)]
res = [
    ('Four-level RTFP (quadratic)', ae4, mk4, 'quadratic', 10),
    ('L1-L6 limitations (quadratic)', aeL, mkL, 'quadratic', 6),
    ('L1-L6 limitations (linear)', aeL, mkL, 'linear', 6),
    ('Independent reviewer L1-L6 (quadratic)', aeH, hyL, 'quadratic', 6),
]
for name, a, b, w, gs in res:
    k, lo, hi = kappa_clustered_bootstrap_ci(a, b, gs, weights=w)
    lines.append(f'{name}: n = {len(a)}; kappa = {k:.4f} [{lo:.4f} - {hi:.4f}]')
lines.append('')
lines.append('CI: 2000-iteration clustered bootstrap resampling whole studies, seed 42.')
open('agreement_eligible_set_results.txt', 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
