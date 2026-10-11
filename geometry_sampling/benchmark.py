"""Multidimensional paired comparison at an equal client-block-gradient budget.

Independent of experiments.py/verify.py: this entry never overwrites results.json.
The gradient-table method is a client-component control-variate estimator, not
SCAFFOLD or an implementation of the original shared-variable SAGA algorithm.
No diagnostic gradient is reused by training or by the gradient table.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
from pathlib import Path
import platform
import time

import numpy as np


HERE = Path(__file__).resolve().parent
METHODS = ("geometric", "inverse_inclusion", "client_gradient_table")
LABELS = {"geometric": "Geometric", "inverse_inclusion": "Inverse inclusion",
          "client_gradient_table": "Gradient table"}
ZERO_TOL = 1e-10
REPEATS = 20
N, B, Q, M = 32, 4, 8, 8
TRAINING_BUDGET = 8000
SELECTED_STEP = 0.015
SEED_BASES = {"common_potential_nonconvex": (51000, 61000),
              "cross_block_nonconvex_stress": (52000, 62000)}


def center(u):
    return u - u.mean(axis=0, keepdims=True)


def norm2(u):
    return float(np.mean(np.sum(u * u, axis=tuple(range(1, u.ndim)))))


def block_radii(u):
    return np.sqrt(np.mean(np.sum(center(u) ** 2, axis=2), axis=0))


def prox_block(y, h, tau):
    """Radial shrinkage in the average product metric, for vector blocks."""
    w = y.mean(axis=0, keepdims=True)
    d = y - w
    radius = float(np.sqrt(np.mean(np.sum(d * d, axis=1))))
    alpha = max(0.0, 1.0 - h * tau / radius) if radius else 0.0
    return w + alpha * d


def prox_all(y, h, tau):
    w = y.mean(axis=0, keepdims=True)
    d = y - w
    radii = np.sqrt(np.mean(np.sum(d * d, axis=2), axis=0))
    ratio = np.divide(h * tau, radii, out=np.full_like(radii, np.inf),
                      where=radii > 0.0)
    return w + np.maximum(0.0, 1.0 - ratio)[None, :, None] * d


def estimate(g, ids, n, method, table_block=None):
    """Use only M current gradients, and the pre-draw cache for the table method."""
    m = len(ids)
    if method == "geometric":
        mean = g.mean(axis=0)
        result = np.broadcast_to(mean, (n, g.shape[1])).copy()
        result[ids] += (n - 1) / (m - 1) * (g - mean)
    elif method == "inverse_inclusion":
        result = np.zeros((n, g.shape[1]))
        result[ids] = n / m * g
    elif method == "client_gradient_table":
        # E[H_i + (N/M) I_i (g_i-H_i) | current state, old H] = g_i.
        result = table_block.copy()
        result[ids] += n / m * (g - table_block[ids])
    else:
        raise ValueError(method)
    return result


@dataclass
class Model:
    name: str
    empirical: np.ndarray
    population: np.ndarray
    theta: np.ndarray
    tau: np.ndarray
    coupling: np.ndarray
    observations_sha256: str
    amplitude: float = 5.0
    samples_per_client: int = 32
    noise_sd: float = 0.04
    initial: float = 3.6

    @property
    def L(self):
        return 1.0 + self.amplitude + float(np.linalg.norm(self.coupling, ord=2))

    def initial_state(self):
        return np.full_like(self.empirical, self.initial)

    def sampled_gradient(self, u, ids, k):
        # The only training-gradient path. Coupled losses inspect other blocks,
        # but compute only the selected clients' selected block derivatives.
        x = u[ids, k]
        result = x + self.amplitude * np.sin(x) - self.empirical[ids, k]
        if np.any(self.coupling[k]):
            result += np.einsum("b,ibq->iq", self.coupling[k], u[ids])
        return result

    def full_gradient(self, u):
        return (u + self.amplitude * np.sin(u) - self.empirical
                + np.einsum("bc,icq->ibq", self.coupling, u))

    def potential(self, u):
        return (0.5 * np.sum(u * u, axis=(1, 2))
                + self.amplitude * np.sum(1.0 - np.cos(u), axis=(1, 2))
                + 0.5 * np.einsum("ibq,bc,icq->i", u, self.coupling, u))

    def objective(self, u):
        return float(np.mean(self.potential(u)
                              - np.sum(self.empirical * u, axis=(1, 2)))
                     + self.tau @ block_radii(u))

    def paired_population_gain(self, u):
        w = np.broadcast_to(u.mean(axis=0, keepdims=True), u.shape)
        return float(np.mean(self.potential(w) - self.potential(u)
                              + np.sum(self.population * (u - w), axis=(1, 2))))

    def metadata(self):
        return dict(name=self.name, N=N, B=B, block_dimension=Q,
                    total_dimension=B * Q, amplitude=self.amplitude,
                    smoothness_bound_L=self.L, selected_step=SELECTED_STEP,
                    selected_step_sufficient_limit_at_mu_half_L=1.0 / (9.0 * self.L),
                    initial_scalar=self.initial, samples_per_client=self.samples_per_client,
                    observation_noise_sd=self.noise_sd, tau=self.tau.tolist(),
                    coupling_matrix=self.coupling.tolist(),
                    empirical_mean=self.empirical.tolist(),
                    population_mean=self.population.tolist(), theta=self.theta.tolist(),
                    observations_sha256=self.observations_sha256,
                    raw_observation_scalar_reads=N * self.samples_per_client * B * Q,
                    sharing_labels=[True, True, False, False],
                    structure_label_interpretation=(
                        "Zero/nonzero population linear-tilt blocks; shared signals have "
                        "identical population losses in the separable common potential."
                        if not np.any(self.coupling) else
                        "Zero/nonzero population linear-tilt blocks only. Cross-block "
                        "coupling transfers heterogeneous signals, so these labels do "
                        "not identify the sharing structure of a population minimizer."))


def make_model(name, data_seed):
    rng = np.random.default_rng(data_seed)
    theta = np.zeros((N, B, Q))
    for b, size in ((2, 0.85), (3, 1.2)):
        v = center(rng.normal(size=(N, Q)))
        theta[:, b] = size * v / np.sqrt(norm2(v))
    population = 0.4 + theta
    observations = population[:, None, :, :] + 0.04 * rng.normal(size=(N, 32, B, Q))
    coupling = np.zeros((B, B))
    if name == "cross_block_nonconvex_stress":
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
            coupling[a, b] = coupling[b, a] = 0.4
    tau = np.array([0.15, 0.15, 0.25, 0.30])
    return Model(name, observations.mean(axis=1), population, theta, tau, coupling,
                 hashlib.sha256(observations.tobytes()).hexdigest())


def full_mapping(u, model):
    mu = 1.0 / (2.0 * model.L)
    return norm2((u - prox_all(u - mu * model.full_gradient(u), mu, model.tau)) / mu)


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return dict(mean=values.mean(axis=0).tolist(),
                sd=values.std(axis=0, ddof=1).tolist())


def paired_difference(values):
    values = np.asarray(values, dtype=float)
    answer = mean_sd(values)
    # Two-sided Student-t interval, df=19; descriptive and not multiplicity adjusted.
    half = 2.093024054408263 * values.std(axis=0, ddof=1) / np.sqrt(len(values))
    answer["paired_mean_ci95_lower"] = (values.mean(axis=0) - half).tolist()
    answer["paired_mean_ci95_upper"] = (values.mean(axis=0) + half).tolist()
    return answer


def local_checks():
    """Check vector geometry, table unbiasedness, and coupled analytic gradients."""
    rng = np.random.default_rng(70401)
    n, m, q = 5, 2, 3
    g = rng.normal(size=(n, q))
    table = rng.normal(size=(n, q))
    means = {method: [] for method in METHODS}
    errors = []
    for subset in combinations(range(n), m):
        ids = np.asarray(subset)
        for method in METHODS:
            v = estimate(g[ids], ids, n, method, table)
            means[method].append(v)
            if method == "geometric":
                errors.append(norm2(v - g))
    unbiasedness = {method: float(np.max(np.abs(np.mean(rows, axis=0) - g)))
                    for method, rows in means.items()}
    coefficient = (n - m) / (m * (n - 1)) + (n - m) / (m - 1)
    variance_error = abs(float(np.mean(errors)) - coefficient * norm2(center(g)))
    table_formula = (n - m) / m * norm2(g - table)
    table_variance_error = abs(float(np.mean([norm2(v - g) for v in means["client_gradient_table"]]))
                               - table_formula)
    common = np.broadcast_to(rng.normal(size=q), (n, q))
    exact_common = max(float(np.max(np.abs(estimate(common[list(s)], np.asarray(s), n,
                                                    "geometric") - common)))
                       for s in combinations(range(n), m))
    full_participation = max(float(np.max(np.abs(estimate(g, np.arange(n), n, method, table) - g)))
                             for method in METHODS)
    model = make_model("cross_block_nonconvex_stress", 70901)
    u = rng.normal(size=(N, B, Q))
    sampled_match = max(float(np.max(np.abs(model.sampled_gradient(u, np.arange(N), b)
                                            - model.full_gradient(u)[:, b]))) for b in range(B))
    i, b, k, eps = 3, 2, 4, 1e-6
    positive, negative = u.copy(), u.copy()
    positive[i, b, k] += eps
    negative[i, b, k] -= eps
    fi_positive = model.potential(positive)[i] - np.sum(model.empirical[i] * positive[i])
    fi_negative = model.potential(negative)[i] - np.sum(model.empirical[i] * negative[i])
    finite_difference_error = abs(float((fi_positive - fi_negative) / (2 * eps)
                                       - model.full_gradient(u)[i, b, k]))
    y = rng.normal(size=(N, B, Q))
    block_vs_full = float(np.max(np.abs(prox_all(y, 0.07, model.tau)
                                        - np.stack([prox_block(y[:, b], 0.07, model.tau[b])
                                                    for b in range(B)], axis=1))))
    assert max(unbiasedness.values()) < 2e-14
    assert max(variance_error, table_variance_error, exact_common, full_participation,
               sampled_match, block_vs_full) < 5e-13
    assert finite_difference_error < 1e-7
    assert np.linalg.eigvalsh(np.eye(B) + model.coupling).min() > 0.0
    # Equation (variance_stepsize), with p_b=delta=1/B and mu=1/(2L),
    # gives selected step h=gamma/p_b <= 1/(9L).
    assert SELECTED_STEP <= 1.0 / (9.0 * model.L)
    return dict(unbiasedness_max_errors=unbiasedness, geometric_variance_error=variance_error,
                table_variance_error=table_variance_error, common_gradient_exactness_error=exact_common,
                full_participation_error=full_participation,
                sampled_vs_full_gradient_error=sampled_match,
                centered_prox_block_vs_full_error=block_vs_full,
                coupled_gradient_finite_difference_error=finite_difference_error)


def run_method(model, method, blocks, subsets):
    u = model.initial_state()
    init_calls = N * B if method == "client_gradient_table" else 0
    steps = (TRAINING_BUDGET - init_calls) // M
    assert init_calls + M * steps == TRAINING_BUDGET
    table = None
    if init_calls:
        # All N*B client-component derivatives are charged to the training budget.
        # This explicitly uses the training path, not any full-mapping diagnostic.
        table = np.stack([model.sampled_gradient(u, np.arange(N), b)
                          for b in range(B)], axis=1)
    table_bytes = int(table.nbytes) if table is not None else 0
    shared = np.array([True, True, False, False])
    sum_mapping = 0.0
    first_mapping = full_mapping(u, model)
    diagnostic_evaluations = 1
    split_counts = np.zeros(B)
    fusion_counts = np.zeros(B)
    ever_split = np.zeros(B, dtype=bool)
    max_radius = np.zeros(B)
    trace = []
    checkpoint = set(range(0, steps + 1, 25)) | {steps}
    mapping = first_mapping
    for t in range(steps + 1):
        if t:
            mapping = full_mapping(u, model)
            diagnostic_evaluations += 1
        radii = block_radii(u)
        split = (radii > ZERO_TOL) & shared
        fusion = (radii <= ZERO_TOL) & ~shared
        ever_split |= split
        max_radius = np.maximum(max_radius, radii)
        if t in checkpoint:
            cost = init_calls + M * t
            # Running mapping is the uniform-update-prefix diagnostic. The
            # budget version additionally charges initialization at the unchanged
            # initial state, for a like-for-like time average in oracle work.
            trace.append(dict(updates=t, training_client_block_oracles=cost,
                              initialization_client_block_oracles=init_calls,
                              instantaneous_mapping=mapping,
                              running_mapping=sum_mapping / t if t else mapping,
                              budget_running_mapping=(init_calls * first_mapping + M * sum_mapping) / cost
                              if cost else mapping,
                              objective=model.objective(u),
                              shared_false_split_rate=float(np.mean(split[shared])),
                              nonshared_false_fusion_rate=float(np.mean(fusion[~shared])),
                              any_shared_false_split_rate=float(np.mean(ever_split[shared])),
                              radii=radii.tolist()))
        if t == steps:
            break
        sum_mapping += mapping
        split_counts += split
        fusion_counts += fusion
        b, ids = int(blocks[t]), subsets[t]
        current_g = model.sampled_gradient(u, ids, b)
        old_block = None if table is None else table[:, b]
        estimator = estimate(current_g, ids, N, method, old_block)
        if table is not None:
            # Refresh only after forming the estimator with the old table.
            table[ids, b] = current_g
        u[:, b] = prox_block(u[:, b] - SELECTED_STEP * estimator, SELECTED_STEP, model.tau[b])
        assert np.all(np.isfinite(u))
    assert diagnostic_evaluations == steps + 1
    final_radii = block_radii(u)
    answer = dict(mean_squared_full_mapping=sum_mapping / steps,
                  budget_mean_squared_full_mapping=(init_calls * first_mapping + M * sum_mapping)
                  / TRAINING_BUDGET,
                  final_squared_full_mapping=mapping, final_objective=model.objective(u),
                  paired_population_gain=model.paired_population_gain(u),
                  any_false_split=ever_split.tolist(),
                  final_false_split=((final_radii > ZERO_TOL) & shared).tolist(),
                  final_false_fusion=((final_radii <= ZERO_TOL) & ~shared).tolist(),
                  any_shared_false_split_rate=float(np.mean(ever_split[shared])),
                  final_shared_false_split_rate=float(np.mean((final_radii > ZERO_TOL)[shared])),
                  final_nonshared_false_fusion_rate=float(np.mean((final_radii <= ZERO_TOL)[~shared])),
                  random_prefix_shared_false_split_rate=float(np.mean((split_counts / steps)[shared])),
                  random_prefix_nonshared_false_fusion_rate=float(np.mean((fusion_counts / steps)[~shared])),
                  random_prefix_false_split_fraction=(split_counts / steps).tolist(),
                  random_prefix_false_fusion_fraction=(fusion_counts / steps).tolist(),
                  final_radii=final_radii.tolist(), max_block_radius=max_radius.tolist(),
                  final_state=u.tolist(), trace=trace, completed_updates=steps,
                  training_client_block_oracles=TRAINING_BUDGET,
                  initialization_client_block_oracles=init_calls,
                  sampled_update_client_block_oracles=M * steps,
                  gradient_table_entries=int(table.size) if table is not None else 0,
                  gradient_table_bytes=table_bytes,
                  diagnostic_client_block_oracles=N * B * diagnostic_evaluations,
                  diagnostic_full_gradient_evaluations=diagnostic_evaluations)
    return answer


NUMERIC_KEYS = ("mean_squared_full_mapping", "budget_mean_squared_full_mapping",
                "final_squared_full_mapping", "final_objective", "paired_population_gain",
                "any_shared_false_split_rate", "final_shared_false_split_rate",
                "final_nonshared_false_fusion_rate", "random_prefix_shared_false_split_rate",
                "random_prefix_nonshared_false_fusion_rate", "final_radii",
                "random_prefix_false_split_fraction", "random_prefix_false_fusion_fraction")
TRACE_KEYS = ("instantaneous_mapping", "running_mapping", "budget_running_mapping", "objective",
              "shared_false_split_rate", "nonshared_false_fusion_rate", "any_shared_false_split_rate", "radii")


def summarize(runs):
    answer = {}
    for method in METHODS:
        answer[method] = {key: mean_sd([run["methods"][method][key] for run in runs])
                          for key in NUMERIC_KEYS}
        for key in ("any_false_split", "final_false_split", "final_false_fusion"):
            answer[method][key] = mean_sd([run["methods"][method][key] for run in runs])
        answer[method]["trace"] = []
        for j, point in enumerate(runs[0]["methods"][method]["trace"]):
            item = {key: point[key] for key in
                    ("updates", "training_client_block_oracles", "initialization_client_block_oracles")}
            item.update({key: mean_sd([run["methods"][method]["trace"][j][key] for run in runs])
                         for key in TRACE_KEYS})
            answer[method]["trace"].append(item)
    for baseline in METHODS[1:]:
        answer["paired_geometric_minus_" + baseline] = {
            key: paired_difference([np.asarray(run["methods"]["geometric"][key])
                                     - np.asarray(run["methods"][baseline][key]) for run in runs])
            for key in NUMERIC_KEYS}
    return answer


def build_results():
    checks = local_checks()
    cases = []
    for name, (data_base, algorithm_base) in SEED_BASES.items():
        print(f"Running {name}: {REPEATS} paired data/update seeds, equal budget {TRAINING_BUDGET}", flush=True)
        runs = []
        for j in range(REPEATS):
            data_seed, algorithm_seed = data_base + j, algorithm_base + j
            model = make_model(name, data_seed)
            rng = np.random.default_rng(algorithm_seed)
            length = TRAINING_BUDGET // M
            blocks = rng.choice(B, size=length, p=np.full(B, 1.0 / B))
            subsets = np.stack([rng.choice(N, M, replace=False) for _ in range(length)])
            schedule_hash = hashlib.sha256(blocks.tobytes() + subsets.tobytes()).hexdigest()
            methods = {method: run_method(model, method, blocks, subsets) for method in METHODS}
            assert {value["training_client_block_oracles"] for value in methods.values()} == {TRAINING_BUDGET}
            runs.append(dict(data_seed=data_seed, algorithm_seed=algorithm_seed,
                             participation_sequence_sha256=schedule_hash,
                             model=model.metadata(), methods=methods))
            if (j + 1) % 5 == 0:
                print(f"  completed {j + 1}/{REPEATS}", flush=True)
        cases.append(dict(name=name,
                          dimension=dict(N=N, B=B, block_dimension=Q, total_dimension=B * Q),
                          M=M, repetitions=REPEATS, total_training_client_block_budget=TRAINING_BUDGET,
                          selected_step=SELECTED_STEP, gamma=SELECTED_STEP / B,
                          block_probabilities=[1.0 / B] * B,
                          runs=runs, summary=summarize(runs)))
    return dict(schema_version=1, experiment="equal_budget_multidimensional_benchmark",
                methods=list(METHODS), method_labels=LABELS, repeats=REPEATS,
                zero_radius_tolerance=ZERO_TOL, implementation_checks=checks,
                protocol=dict(
                    objective="F_i(u)=0.5||u||^2+5 sum(1-cos(u))+0.5<u,Cu>-<empirical_mean_i,u>; r=sum_b tau_b||DU_b||_N",
                    population_loss="ell_i(u;z)=h(u)-<z,u>; E[z]=population_mean_i; Gaussian SD .04, 32 samples/client",
                    same_objective_initial_data_and_draws=True,
                    exact_current_selected_gradients=True, full_selected_block_prox=True,
                    table_estimator="H_{i,b}+(N/M) 1{i in S}(g_{i,b}(U)-H_{i,b}); refresh sampled table entries after estimation",
                    table_scope="Client-component gradient table estimator; neither SCAFFOLD nor original shared-variable SAGA",
                    equal_budget="Each method spends 8000 client-block gradient calls. Geometric/HT: 1000 updates. Table: 128 full-table initialization calls plus 984 updates*8 calls. Table uses the same first 984 draws of the common 1000-draw schedule.",
                    memory="Table adds N*B*block_dimension=1024 float64 entries=8192 bytes, excluding state/method-independent arrays. Dense reference, no measured communication reduction.",
                    output="Uniform random pre-update prefix for each method at its available update count; final iterate is a separate diagnostic",
                    running_mapping="Exact mean over pre-update full squared proximal mappings. These are deterministic full mappings, not the Moreau-envelope gradient.",
                    budget_running_mapping="Full-table initialization consumes 128 calls at the unchanged initial mapping; subsequent pre-update mappings weighted by M=8 calls. Average over equal total oracle work.",
                    full_mapping_parameter="mu=1/(2L), Hilbert average-client metric; L=6 for separable case, L=6.8 for coupled case",
                    diagnostics="N*B*(T_method+1) extra client-block derivative diagnostics; excluded from training budget, never supplied to any training estimator/cache. Objective/population evaluation also external diagnostics.",
                    preprocessing="32768 scalar observations generated/read per data run; empirical linear coefficients are sufficient statistics, identically available to all methods. No uncharged training-gradient-table initialization.",
                    randomization="20 independent data/update seed pairs per case; methods paired on both seeds and on common schedule prefix; independent data and algorithm RNGs",
                    error_bars="Sample SD across runs; paired-mean descriptive 95% t intervals use df=19, no multiple-comparison correction",
                    structure="First two blocks have zero population tilts. Coupled-case structure metrics concern these signal labels, not population-minimizer sharing labels. Fusion tolerance 1e-10.",
                    limitations="Synthetic exact-gradient dense NumPy reference; fixed equal step, no per-method tuning, no application/communication/large-model speed claim; common noisy objective and strong nonconvex initialization only two controlled cases."),
                cases=cases)


def numeric_sha256(result):
    return hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def write_outputs(result, elapsed, replay_elapsed=None):
    result["execution"] = dict(python=platform.python_version(), numpy=np.__version__,
                               elapsed_wall_seconds=elapsed,
                               numerical_result_sha256=numeric_sha256(result),
                               complete_numerical_replay_checked=replay_elapsed is not None,
                               replay_elapsed_wall_seconds=replay_elapsed,
                               source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE / "benchmark_results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                                   encoding="utf-8")
    with (HERE / "benchmark_results.csv").open("w", encoding="utf-8", newline="") as fp:
        fields = ["case", "data_seed", "algorithm_seed", "method", "completed_updates",
                  "training_client_block_oracles", "initialization_client_block_oracles",
                  "gradient_table_bytes", "diagnostic_client_block_oracles"] + list(NUMERIC_KEYS[:10])
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for case in result["cases"]:
            for run in case["runs"]:
                for method, record in run["methods"].items():
                    row = dict(case=case["name"], data_seed=run["data_seed"],
                               algorithm_seed=run["algorithm_seed"], method=method)
                    row.update({key: record[key] for key in fields[4:]})
                    writer.writerow(row)
    lines = ["Equal-budget multidimensional synthetic benchmark",
             f"N={N}, B={B}, block dimension={Q}, total dimension={B*Q}, M={M}, repeats={REPEATS}.",
             f"Budget={TRAINING_BUDGET} client-block gradients/method/run. Selected step={SELECTED_STEP}.",
             "Gradient table: 128 paid initialization calls + 984*8 update calls; 8192 bytes additional gradient storage.",
             "Geometric and inverse inclusion: 1000*8 update calls. Shared common schedule prefix.",
             "All reported +/- entries are sample SD over independent paired data/update runs.",
             "Budget mean mapping includes initialization cost at the initial state; update-time mean mapping does not.",
             "Structure in the coupled case is measured against linear-tilt signal labels, not minimizer sharing.",
             "Population gain = population risk(broadcast output mean)-population risk(personalized output).",
             f"Actual generation wall time: {elapsed:.3f} seconds.",
             f"Full numerical replay wall time: {replay_elapsed:.3f} seconds." if replay_elapsed is not None
             else "Full numerical replay not requested.", ""]
    for case in result["cases"]:
        lines.append(case["name"])
        for method in METHODS:
            lines.append(LABELS[method])
            for key in NUMERIC_KEYS[:10]:
                value = case["summary"][method][key]
                lines.append(f"  {key}: {value['mean']:.8g} +/- {value['sd']:.8g}")
        for baseline in METHODS[1:]:
            lines.append("Paired geometric minus " + baseline)
            for key in ("budget_mean_squared_full_mapping", "final_squared_full_mapping", "final_objective"):
                value = case["summary"]["paired_geometric_minus_" + baseline][key]
                lines.append(f"  {key}: {value['mean']:.8g}, descriptive 95% paired-mean CI "
                             f"[{value['paired_mean_ci95_lower']:.8g}, {value['paired_mean_ci95_upper']:.8g}]")
        lines.append("")
    lines += ["Limitations: two synthetic exact-gradient cases, fixed common step and thresholds; no per-method",
              "tuning or application/system timing comparison. Dense implementation; memory above counts",
              "gradient-table payload only. Full diagnostics and data preprocessing are separated from",
              "training budget. No blanket superiority or guarantee beyond the manuscript hypotheses.",
              "Numerical checks: " + json.dumps(result["implementation_checks"], sort_keys=True)]
    (HERE / "benchmark_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-reproducibility", action="store_true",
                        help="Run the complete 40-data-pair experiment twice and compare numerical hashes")
    args = parser.parse_args()
    started = time.perf_counter()
    result = build_results()
    elapsed = time.perf_counter() - started
    replay_elapsed = None
    if args.check_reproducibility:
        first_hash = numeric_sha256(result)
        replay_started = time.perf_counter()
        repeated = build_results()
        replay_elapsed = time.perf_counter() - replay_started
        assert first_hash == numeric_sha256(repeated), "Complete numerical replay failed"
    write_outputs(result, elapsed, replay_elapsed)
    print(f"Saved benchmark_results.json/csv and benchmark_summary.txt; wall time {elapsed:.3f}s", flush=True)
    if replay_elapsed is not None:
        print(f"Complete numerical replay matched; replay wall time {replay_elapsed:.3f}s", flush=True)


if __name__ == "__main__":
    main()
