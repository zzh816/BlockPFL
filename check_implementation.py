"""Run small algebra/gradient checks and verify the shipped archive checksums.

No simulations are rerun and no archived outputs are overwritten.
"""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent
GEO = ROOT / "geometry_sampling"
sys.path.insert(0, str(GEO))
sys.path.insert(0, str(GEO / "allocation_revision"))
import experiments
import benchmark
import allocation_experiment
import wdbc_experiment


def main():
    expected = json.loads((ROOT / "ARCHIVE_SHA256.json").read_text(encoding="utf-8"))
    for name, digest in expected.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    result = {
        "archive_files_checked": len(expected),
        "exact_variance_cases": len(experiments.exact_variance_checks()),
        "benchmark": benchmark.local_checks(),
        "allocation": allocation_experiment.local_checks(),
        "wdbc": wdbc_experiment.checks(wdbc_experiment.make_partition(81000)),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
