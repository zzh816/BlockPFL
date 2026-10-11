# Data provenance and attribution

## Wisconsin Diagnostic Breast Cancer (WDBC)

**Citation:** Wolberg, W., Mangasarian, O., Street, N., and Street, W. (1993). *Breast Cancer Wisconsin (Diagnostic)* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B).

- [Official UCI dataset page](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- [Original data file](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data)
- License: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- Repository copy: `geometry_sampling/allocation_revision/wdbc.data`
- SHA256: `d606af411f3e5be8a317a5a8b652b425aaf0ff38ca683d5327ffff94c3695f4a`

The repository stores the original CSV-format data file **without modification**. It contains 569 rows, a case identifier, a diagnosis label, and 30 numerical features. The script excludes the case identifier from the model, maps malignant (`M`) to 1 and benign (`B`) to 0, and checks the file hash before use. Public dataset identifiers are source case identifiers, not measurements of user or agent behavior.

For each split, `wdbc_experiment.py` fits standardization on the training rows only and appends an intercept in memory. Training/test assignments, fitted means and standard deviations, client row assignments, and client sample sizes are saved in `wdbc_results.json`; these transformations do not overwrite the original file. Split seeds are 81000–81019 and independent update seeds are 91000–91019.

The eight “clients” are artificially generated label-skew partitions of the public observations. They are not actual users, agents, hospitals, or a deployment cohort. Both training and test pools are partitioned separately using the stated skew probabilities. No clinical efficacy, true client-sharing labels, or structural identification claim is inferred from this dataset. See [EXPERIMENTS.md](EXPERIMENTS.md#43-coupled-logistic-objective-on-wdbc) for the full split, preprocessing, objective, and evaluation protocol.

The attribution and CC BY 4.0 link above apply to the WDBC data. They do not assign a license to the separately authored experiment code.

## Synthetic observations

Synthetic experiments generate Gaussian observations and deterministic or rescaled centered tilts using the seeds documented in [EXPERIMENTS.md](EXPERIMENTS.md). These observations are not records of real people. Numerical archives retain parameters, seeds, observation hashes or sufficient statistics, and per-run outcomes. The original scalar-block nonconvex observations are additionally preserved in `geometry_sampling/fixtures/mechanism_reference.json`.

