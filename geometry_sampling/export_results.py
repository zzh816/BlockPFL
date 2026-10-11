"""Export archived experiments to per-run long CSV and a unified text summary.

Only the Python standard library is used; no simulations or plotting are run.
Benchmark statistics are recomputed from benchmark_results.csv and cross-checked
against benchmark_results.json before writing output.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
T_CRIT_19 = 2.093024054408263
NORMAL_975 = 1.959963984540054
ORIGINAL_SCALARS = ("mean_squared_full_mapping", "final_squared_full_mapping",
                    "final_objective", "mean_conditional_estimator_variance",
                    "paired_population_gain")
BENCHMARK_COMPARISONS = ("budget_mean_squared_full_mapping",
                         "final_squared_full_mapping", "final_objective")
META_FIELDS = ["family", "case", "run_index", "data_seed", "algorithm_seed", "method",
               "N", "B", "M", "T", "gamma", "block_probabilities", "tau", "model_kind",
               "coupling", "training_client_block_oracles", "diagnostic_client_block_oracles",
               "participation_sequence_sha256", "source_sha256", "zero_radius_tolerance"]


def load_json(path):
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def groups(result):
    for family in ("main_groups", "tau_sweep", "noise_participation_sweep"):
        for group in result[family]:
            yield family, group
    yield "coupled_group", result["coupled_group"]


def leaves(value, component=""):
    """Yield numeric array leaves; component uses zero-based bracket indices."""
    if isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaves(item, f"{component}[{index}]")
    elif isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Nonfinite archived value")
        yield component, number


def metrics(record, prefix=""):
    for key, value in record.items():
        if key == "trace":
            continue
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from metrics(value, name)
        else:
            for component, number in leaves(value):
                yield name, component, number


def metadata(family, group, index, run, method, record, digest, tolerance):
    model = group["models"][index]
    return dict(family=family, case=group["name"], run_index=index,
                data_seed=run["data_seed"], algorithm_seed=run["algorithm_seed"], method=method,
                N=model["N"], B=model["B"], M=group["M"], T=group["T"], gamma=group["gamma"],
                block_probabilities=json.dumps(group["block_probabilities"], separators=(",", ":")),
                tau=json.dumps(model["tau"], separators=(",", ":")), model_kind=model["kind"],
                coupling=model["coupling"], training_client_block_oracles=record["training_client_block_oracles"],
                diagnostic_client_block_oracles=record["diagnostic_client_block_oracles"],
                participation_sequence_sha256=run["participation_sequence_sha256"],
                source_sha256=digest, zero_radius_tolerance=tolerance)


def export_original(result, digest, output):
    count, traces = 0, 0
    fields = META_FIELDS + ["metric", "component", "value"]
    with (output / "original_run_metrics_long.csv").open("w", encoding="utf-8", newline="") as fp, \
         (output / "original_trace_metrics_long.csv").open("w", encoding="utf-8", newline="") as tp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        trace_writer = csv.DictWriter(tp, fieldnames=META_FIELDS + ["updates", "training_oracles",
                                                                  "metric", "component", "value"])
        writer.writeheader()
        trace_writer.writeheader()
        for family, group in groups(result):
            if len(group["runs"]) != 20 or group["repetitions"] != 20:
                raise ValueError("This archived protocol requires 20 paired runs per configuration")
            for index, run in enumerate(group["runs"]):
                for method, record in run["methods"].items():
                    base = metadata(family, group, index, run, method, record, digest,
                                    result["zero_radius_tolerance"])
                    for metric, component, value in metrics(record):
                        writer.writerow(dict(base, metric=metric, component=component, value=value))
                        count += 1
                    for checkpoint in record["trace"]:
                        for metric, component, value in metrics(checkpoint):
                            if metric in ("updates", "training_oracles"):
                                continue
                            trace_writer.writerow(dict(base, updates=checkpoint["updates"],
                                                       training_oracles=checkpoint["training_oracles"],
                                                       metric=metric, component=component, value=value))
                            traces += 1
    return count, traces


def mean_interval(values):
    if len(values) != 20:
        raise ValueError("The df=19 t interval requires exactly 20 observations")
    mean, sd = statistics.mean(values), statistics.stdev(values)
    half = T_CRIT_19 * sd / math.sqrt(len(values))
    return mean, sd, mean - half, mean + half


def fmt_interval(values):
    mean, sd, lower, upper = mean_interval(values)
    return f"{mean:.8g} +/- {sd:.8g}; mean CI95 [{lower:.8g}, {upper:.8g}]"


def wilson(events):
    n, successes = len(events), sum(bool(value) for value in events)
    p, z2 = successes / n, NORMAL_975 ** 2
    center = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = NORMAL_975 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return successes, n, max(0.0, center - half), min(1.0, center + half)


def original_summary(result):
    lines = ["ORIGINAL TWO-SAMPLER EXPERIMENTS",
             "20 paired runs/configuration; fixed-data cases vary update RNG only; noise sweep varies data/update RNG.",
             "Mean mapping = exact uniform pre-update prefix mean squared FULL proximal mapping at mu=1/(2L).",
             "Final mapping/objective/population gain are separate final-state diagnostics.",
             "Training cost is M*T. Diagnostic derivatives are separately charged N*B*(T+2); raw sample preprocessing is separate.",
             "Conditional estimator variance uses sum p_b*sigma_b^2, not the theorem's inverse-probability noise budget.",
             "Numerical structure uses centered-radius tolerance 1e-10, not exact-zero theorem events.", ""]
    for family, group in groups(result):
        model = group["models"][0]
        lines.append(f"{family}/{group['name']}: N={model['N']}, B={model['B']}, M={group['M']}, "
                     f"T={group['T']}, gamma={group['gamma']:.8g}, M*T={group['M'] * group['T']}, tau={model['tau']}")
        shared = [b for b in range(model["B"])
                  if all(abs(row[b]) <= 1e-14 for row in model["theta"])]
        for method in result["methods"]:
            records = [run["methods"][method] for run in group["runs"]]
            lines.append("  " + method)
            for key in ORIGINAL_SCALARS:
                label = "final_population_gain" if key == "paired_population_gain" else key
                lines.append(f"    {label}: {fmt_interval([record[key] for record in records])}")
            if shared:
                events = [any(record["any_false_split"][b] for b in shared) for record in records]
                successes, n, lower, upper = wilson(events)
                lines.append(f"    any numerical split on labeled shared blocks: {successes}/{n}; "
                             f"Wilson CI95 [{lower:.8g}, {upper:.8g}]")
        differences = [run["methods"]["geometric"]["mean_squared_full_mapping"]
                       - run["methods"]["inverse_inclusion"]["mean_squared_full_mapping"]
                       for run in group["runs"]]
        lines.append("  Paired geometric minus inverse inclusion, prefix mean mapping: " + fmt_interval(differences))
        lines.append("")
    lines += ["Coupled quadratic: fixed public geometry bound checks are not actual sampled-score measurements",
              "and are not an HT sharing certificate; both methods may remain shared under a conservative threshold.",
              "This certificate instance is a specific positive-definite quadratic subclass, not a general nonconvex extension.", ""]
    return lines


def read_benchmark(csv_path, result):
    csv_bytes = csv_path.read_bytes()
    reader = csv.DictReader(io.StringIO(csv_bytes.decode("utf-8-sig")))
    fields = reader.fieldnames
    if not fields:
        raise ValueError("Empty benchmark CSV")
    rows = list(reader)
    expected = {}
    for case in result["cases"]:
        for run in case["runs"]:
            for method, record in run["methods"].items():
                key = (case["name"], str(run["data_seed"]), str(run["algorithm_seed"]), method)
                expected[key] = record
    found = set()
    for row in rows:
        key = (row["case"], row["data_seed"], row["algorithm_seed"], row["method"])
        if key in found or key not in expected:
            raise ValueError(f"Duplicate or unmatched benchmark CSV row: {key}")
        found.add(key)
        for field in fields[4:]:
            value = float(row[field])
            if not math.isfinite(value) or not math.isclose(value, expected[key][field], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"Benchmark CSV/JSON mismatch: {key}, {field}")
    if found != set(expected):
        raise ValueError("Benchmark CSV is missing archived runs")
    return rows, fields, hashlib.sha256(csv_bytes).hexdigest()


def export_benchmark(rows, fields, result, csv_digest, output):
    lines = ["NEW MULTIDIMENSIONAL GRADIENT-TABLE BENCHMARK"]
    for key in ("equal_budget", "memory", "output", "budget_running_mapping", "diagnostics", "preprocessing",
                "randomization", "table_scope", "structure", "limitations"):
        lines.append(f"{key}: {result['protocol'][key]}")
    lines.append("All statistics below are recomputed from benchmark_results.csv, cross-checked against archived JSON.")
    lines.append("Structure rates are averages over labeled blocks, not necessarily Bernoulli run events.")
    lines.append("paired_population_gain is evaluated at the FINAL state; it is not a random-prefix risk guarantee.")
    lines.append("")
    csv_fields = ["case", "statistic", "method_or_difference", "metric", "n", "mean", "sample_sd",
                  "ci95_lower", "ci95_upper", "interval", "source_csv_sha256"]
    numeric_metrics = fields[9:]
    count = 0
    with (output / "benchmark_summary_export.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=csv_fields)
        writer.writeheader()
        for case in result["cases"]:
            case_rows = [row for row in rows if row["case"] == case["name"]]
            lines.append(case["name"])
            model = case["runs"][0]["model"]
            lines.append(f"  dimension={case['dimension']}; M={case['M']}; repetitions={case['repetitions']}; "
                         f"budget={case['total_training_client_block_budget']}; h={case['selected_step']}; "
                         f"gamma={case['gamma']}; p={case['block_probabilities']}; tau={model['tau']}; "
                         f"L={model['smoothness_bound_L']}; "
                         f"sufficient h limit={model['selected_step_sufficient_limit_at_mu_half_L']:.8g}")
            by_method = {method: [row for row in case_rows if row["method"] == method]
                         for method in result["methods"]}
            for method, method_rows in by_method.items():
                lines.append("  " + result["method_labels"][method])
                for metric in numeric_metrics:
                    values = [float(row[metric]) for row in method_rows]
                    mean, sd, lower, upper = mean_interval(values)
                    writer.writerow(dict(case=case["name"], statistic="method_mean", method_or_difference=method,
                                         metric=metric, n=len(values), mean=mean, sample_sd=sd, ci95_lower=lower,
                                         ci95_upper=upper, interval="nominal_pointwise_95pct_t_df19",
                                         source_csv_sha256=csv_digest))
                    count += 1
                    lines.append(f"    {metric}: {fmt_interval(values)}")
            geometric = {(row["data_seed"], row["algorithm_seed"]): row for row in by_method["geometric"]}
            for baseline in result["methods"][1:]:
                label = "geometric_minus_" + baseline
                lines.append("  Paired " + label)
                for metric in BENCHMARK_COMPARISONS:
                    values = [float(geometric[(row["data_seed"], row["algorithm_seed"])][metric])
                              - float(row[metric]) for row in by_method[baseline]]
                    mean, sd, lower, upper = mean_interval(values)
                    writer.writerow(dict(case=case["name"], statistic="paired_difference", method_or_difference=label,
                                         metric=metric, n=len(values), mean=mean, sample_sd=sd, ci95_lower=lower,
                                         ci95_upper=upper, interval="nominal_95pct_paired_mean_t_df19",
                                         source_csv_sha256=csv_digest))
                    archived = case["summary"]["paired_geometric_minus_" + baseline][metric]
                    for actual, expected in ((mean, archived["mean"]), (lower, archived["paired_mean_ci95_lower"]),
                                             (upper, archived["paired_mean_ci95_upper"])):
                        if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
                            raise ValueError("Recomputed paired interval differs from benchmark archive")
                    count += 1
                    lines.append(f"    {metric}: {fmt_interval(values)}")
            lines.append("")
    return lines, count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results.json")
    parser.add_argument("--benchmark", type=Path, default=HERE / "benchmark_results.json")
    parser.add_argument("--benchmark-csv", type=Path, default=HERE / "benchmark_results.csv")
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    original, original_digest = load_json(args.results)
    benchmark, benchmark_digest = load_json(args.benchmark)
    rows, fields, benchmark_csv_digest = read_benchmark(args.benchmark_csv, benchmark)
    args.output.mkdir(parents=True, exist_ok=True)
    run_count, trace_count = export_original(original, original_digest, args.output)
    benchmark_lines, summary_count = export_benchmark(rows, fields, benchmark, benchmark_csv_digest, args.output)
    replay_path = args.results.with_name("reproducibility.json")
    replay_note = "Original complete numerical replay record not supplied."
    if replay_path.exists():
        replay, _ = load_json(replay_path)
        numeric = replay.get("numerical_replay", {})
        matches = numeric.get("results_sha256") == original_digest
        replay_note = (f"Original numerical replay checked={numeric.get('checked', False)}, "
                       f"identical_bytes={numeric.get('identical_bytes', False)}, source_digest_matches={matches}.")
    header = ["CURRENT-MANUSCRIPT EXPERIMENT SUMMARY (recomputed from archived per-run data)",
              f"Original JSON SHA256: {original_digest}", f"Benchmark JSON SHA256: {benchmark_digest}",
              f"Benchmark CSV SHA256: {benchmark_csv_digest}",
              replay_note,
              f"Benchmark complete numerical replay checked={benchmark['execution']['complete_numerical_replay_checked']}; "
              f"Python={benchmark['execution']['python']}; NumPy={benchmark['execution']['numpy']}.",
              "Original array components use zero-based bracket indices; final_state indices are [client][block].",
              "+/- means sample SD. Continuous mean intervals use t(df=19); paired intervals use run differences.",
              "These are nominal 95% descriptive intervals: no simultaneous coverage or multiple-comparison correction.",
              "Publication plots use per-method pointwise mean intervals; those are not paired-difference intervals.",
              "Binary event intervals are Wilson, and numerical events use radius tolerance 1e-10.",
              "No universal superiority: geometric sampling can lose in zero-mean/high-disagreement regimes,",
              "and the gradient table can give better final residual/objective under paid initialization and additional memory.",
              "Coupled nonconvex labels are linear-tilt signal labels, not population-minimizer sharing labels.",
              "Synthetic exact-gradient dense reference; no image/LLM/agent application or measured communication speed claim.", ""]
    text = "\n".join(header + original_summary(original) + benchmark_lines) + "\n"
    (args.output / "experiment_summary.txt").write_text(text, encoding="utf-8")
    print(f"Exported {run_count} original metric rows, {trace_count} checkpoint rows, "
          f"{summary_count} benchmark summary rows; benchmark CSV/JSON and paired intervals agree.", flush=True)


if __name__ == "__main__":
    main()
