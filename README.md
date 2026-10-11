# Geometry-Consistent Sampling for Nonconvex Personalized Optimization

Companion experiments for the manuscript of the same title.

Repository: [Geometry-Consistent Sampling for Nonconvex Personalized Optimization](https://github.com/zzh816/Geometry-Consistent-Sampling-for-Nonconvex-Personalized-Optimization).

This repository contains the experiments used in the current manuscript: current-gradient sampling and threshold effects, a gradient-table comparison, and uniform/fixed-pilot/adaptive block-probability comparisons on synthetic and WDBC logistic objectives. The numerical results include settings where the geometric or adaptive method performs worse.

## Documentation

- [Experimental protocols and complete settings](docs/EXPERIMENTS.md): objectives, parameters, seeds, pairing, diagnostics, resource accounting, and interpretation of the results. This document contains the experimental details moved out of the manuscript appendix.
- [Dataset provenance](docs/DATA.md): the original UCI WDBC source, attribution, checksum, and preprocessing.
- [Package validation](docs/VALIDATION.md): the checks performed when preparing this standalone release and their scope.

The contents of `codes1010` are intended to form the root of this GitHub repository. The paper's `codes1010/run_experiments.py` is therefore `run_experiments.py` when working in the repository root.

## Environment

The archived numerical runs used Python 3.12.14 and NumPy 2.3.5. Use Python 3.12 and the pinned dependencies for numerical reproduction. Matplotlib is needed for figures. The repository contains source and data; it does not bundle a Python environment.

```sh
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in Windows PowerShell or `source .venv/bin/activate` in a POSIX shell, then run:

```sh
python -m pip install -r requirements_experiments.txt
python run_experiments.py --dry-run
```

## Reproduce

Check the supplied archive checksums and small algebra/gradient identities before regenerating numerical outputs:

```sh
python check_implementation.py
```

The checksums in `ARCHIVE_SHA256.json` describe the supplied archives. Regenerated files can have different source/environment metadata or timing fields even when numerical outcomes agree.

Regenerate the publication figures and CSV summaries from the included numerical results:

```sh
python run_experiments.py --figures-only
```

Run every experiment once, followed by plotting and export:

```sh
python run_experiments.py --skip-replay
```

Run the complete pipeline with the implemented replay checks:

```sh
python run_experiments.py
```

The default pipeline replays the mechanism experiments, the multidimensional benchmark, and the synthetic probability ablation. The WDBC experiment runs once. Timing fields and environment/source metadata are distinguished from numerical reproducibility. Simulation commands regenerate the result files; `--figures-only` reads the existing numerical results. Paths are resolved relative to the scripts, so the entry point also works from another working directory.

## Experiment files

| Experiment | Script under `geometry_sampling/` | Numerical results |
| --- | --- | --- |
| Sampling mechanism, participation, thresholds, noise, and coupled certificate | `verify.py`, `experiments.py` | `results.json` |
| Geometric / inverse-inclusion / gradient-table benchmark | `benchmark.py` | `benchmark_results.json`, `benchmark_results.csv` |
| Synthetic block-probability ablation | `allocation_revision/allocation_experiment.py` | `allocation_revision/allocation_results.json`, `.csv` |
| WDBC logistic block-probability ablation | `allocation_revision/wdbc_experiment.py` | `allocation_revision/wdbc_results.json`, `.csv` |

`plot_publication.py` creates the mechanism and baseline figures in `geometry_sampling/figures/`. `allocation_revision/make_allocation_figure.py` creates `allocation_comparison.png`. `export_results.py` exports run-level and trace-level CSV files and benchmark summaries. The JSON archives contain configurations, seed information, paired outcomes, and cost counters; consult the protocol for differences between experiment families.

## Reading the results

An oracle call is a current local empirical block-gradient query. Pilot and gradient-table initialization queries are charged. Full-gradient diagnostics are recorded separately and never supplied to the sampling rule. A uniform update-prefix average and a cost-prefix average with paid initialization are different measures. Observed first threshold crossings are diagnostic quantities rather than free stopping certificates.

The structural result applies under the manuscript's specified model and certificates. The coupled WDBC logistic experiment has artificial client partitions and no structural ground-truth labels. These experiments do not establish a universal convergence, generalization, communication, or large-model training-speed advantage. See [the complete protocol](docs/EXPERIMENTS.md) for the favorable and unfavorable outcomes and uncertainty summaries.
