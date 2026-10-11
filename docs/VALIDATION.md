# Package validation

The package was checked on 2026-10-12 with Python 3.12.14 and NumPy 2.3.5. Figure generation additionally used Matplotlib 3.11.2.

- The unified entry point resolves all seven pipeline commands from the package directory.
- The 13 supplied numerical/data archives listed in `ARCHIVE_SHA256.json` match their recorded SHA256 values.
- Small implementation checks pass for exact variance and unbiasedness identities, proximal updates, finite-difference gradients, probability scores, and disjoint WDBC training/test splits.
- All three figure/export stages run from a separate working directory in a temporary copy. The supplied numerical archives are not overwritten by that check.
- The first archived synthetic allocation seed and first WDBC seed were each run for all three strategies at their full prescribed budgets. All numerical fields match the archived results exactly after excluding timing fields: six method runs in total.
- An independent source comparison found the numerical functions unchanged. Packaging changes concern paths, dependency loading, the orchestration entry point, and removal of manuscript LaTeX generation.

The full multi-seed suite was not rerun during packaging. The supplied result archives retain their original experiment and replay records. Runtime measurements depend on the machine and are not expected to reproduce exactly.

To check the supplied files before regenerating results, run `python check_implementation.py`. After a simulation run, changed source/environment metadata and timings can change file hashes even when the numerical fields match. `ARCHIVE_SHA256.json` identifies the originally supplied archive, rather than serving as an expected checksum for every future run.
