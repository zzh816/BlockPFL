"""Run the current manuscript experiments, publication plots, and CSV export.

Paths are independent of the calling directory. Child processes use the same
Python executable; no historical image-data prototype is invoked.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE
EXPERIMENTS = HERE / "geometry_sampling"


def commands(figures_only: bool, skip_replay: bool) -> list[list[str]]:
    prefix = [sys.executable, "-X", "utf8"]
    result = []
    if not figures_only:
        verify = prefix + [str(EXPERIMENTS / "verify.py")]
        benchmark = prefix + [str(EXPERIMENTS / "benchmark.py")]
        if not skip_replay:
            verify.append("--check-reproducibility")
            benchmark.append("--check-reproducibility")
        result.extend([verify, benchmark])
        allocation = prefix + [str(EXPERIMENTS / "allocation_revision" / "allocation_experiment.py")]
        if not skip_replay:
            allocation.append("--replay-check")
        result.extend([allocation, prefix + [str(EXPERIMENTS / "allocation_revision" / "wdbc_experiment.py")]])
    result.extend([prefix + [str(EXPERIMENTS / "plot_publication.py")],
                   prefix + [str(EXPERIMENTS / "export_results.py")],
                   prefix + [str(EXPERIMENTS / "allocation_revision" / "make_allocation_figure.py")]])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures-only", action="store_true",
                        help="Regenerate plots and exports from archived results; no simulations/replay")
    parser.add_argument("--skip-replay", action="store_true",
                        help="Generate each numerical experiment once rather than twice")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running or modifying any output")
    args = parser.parse_args()
    steps = commands(args.figures_only, args.skip_replay)
    if args.dry_run:
        for command in steps:
            print(subprocess.list2cmdline(command), flush=True)
        return 0
    if not args.figures_only:
        # Use the pinned numerical dependency for archive reproduction.
        subprocess.run([sys.executable, "-X", "utf8", "-c",
                        "import sys,numpy; print('Numerical Python:',sys.version.split()[0],"
                        "'NumPy:',numpy.__version__,flush=True); "
                        "assert numpy.__version__=='2.3.5',"
                        "'Numerical replay requires NumPy 2.3.5; use requirements_experiments.txt'"],
                       cwd=ROOT, check=True)
    for index, command in enumerate(steps, 1):
        print(f"[{index}/{len(steps)}] {Path(command[3]).name}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    print("Current-manuscript experiment pipeline completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
