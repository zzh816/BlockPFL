"""Generate mechanism results and optionally check complete numerical replay."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import experiments

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-reproducibility", action="store_true")
    args = parser.parse_args()
    result_path = HERE / "results.json"
    experiments.generate()
    first = hashlib.sha256(result_path.read_bytes()).hexdigest()
    if args.check_reproducibility:
        experiments.generate()
        assert first == hashlib.sha256(result_path.read_bytes()).hexdigest(), "Numerical replay differs"
    sources = [HERE / "verify.py", HERE / "experiments.py", HERE / "fixtures/mechanism_reference.json"]
    source_hashes = {str(p.relative_to(HERE)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sources}
    replay = dict(checked=args.check_reproducibility, identical_bytes=args.check_reproducibility,
                  results_sha256=first, source_sha256_at_full_replay=source_hashes)
    record = dict(python_version=sys.version, numpy_version=experiments.np.__version__,
                  mode="full_generation", numerical_replay=replay, source_sha256=source_hashes,
                  output_sha256={"geometry_sampling/results.json": first})
    (HERE / "reproducibility.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("Mechanism results generated; complete numerical replay checked=" + str(args.check_reproducibility))


if __name__ == "__main__":
    main()
