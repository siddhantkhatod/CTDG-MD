import numpy as np

from ctdg_md.statistics import holm_bonferroni, paired_complex_bootstrap


def test_cluster_bootstrap_detects_paired_improvement():
    groups = np.repeat(np.array(["a", "b", "c", "d", "e", "f", "g", "h"]), 10)
    truth = np.linspace(-3, 3, 80)
    baseline = truth + np.tile([1.0, -1.0], 40)
    candidate = truth + 0.1 * np.tile([1.0, -1.0], 40)
    comparison = paired_complex_bootstrap(
        truth,
        candidate,
        baseline,
        groups,
        "candidate",
        "baseline",
        replicates=2000,
        seed=7,
    )
    assert comparison.mae_difference < 0
    assert comparison.ci_high < 0
    adjusted = holm_bonferroni([comparison])[0]
    assert adjusted.holm_adjusted_p is not None
    assert adjusted.holm_adjusted_p < 0.05
