"""Predeclared, paired block-probability ablation for geometric proximal updates.

Only selected current node/block gradients enter training. Full deterministic
proximal mappings are separately counted/timed diagnostics, never training input.
The EMA scores are estimates, NOT certified upper bounds. No tuning or rejection
of unfavorable seeds is performed. Requires NumPy; plotting additionally uses
Matplotlib. Run: python allocation_experiment.py [--replay-check].
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

HERE = Path(__file__).resolve().parent
N, B, Q, M = 32, 4, 8, 8
REPEATS, BUDGET = 20, 32000
DELTA, BETA, ETA = 0.05, 0.1, 1e-12
L, MU = 6.0, 1.0 / 12.0
GAMMA = DELTA / (12.0 * L)
TARGETS = (1.0, 0.1, 0.01)
ZERO_TOL = 1e-10
METHODS = ("uniform", "fixed_pilot", "adaptive_ema")
LABELS = dict(uniform="Uniform", fixed_pilot="Fixed pilot", adaptive_ema="Adaptive EMA")
SHARED = np.array([True, True, False, False])


def center(x):
    return x - x.mean(axis=0, keepdims=True)


def norm2(x):
    return float(np.mean(np.sum(x * x, axis=tuple(range(1, x.ndim)))))


def radii(u):
    return np.sqrt(np.mean(np.sum(center(u) ** 2, axis=2), axis=0))


def prox_block(y, h, tau):
    mean = y.mean(axis=0, keepdims=True)
    difference = y - mean
    radius = np.sqrt(norm2(difference))
    return mean + max(0.0, 1.0 - h * tau / radius) * difference if radius else mean + 0 * difference


def prox_all(y, h, tau):
    mean = y.mean(axis=0, keepdims=True)
    difference = y - mean
    radius = radii(y)
    shrink = np.divide(h * tau, radius, out=np.full_like(radius, np.inf), where=radius > 0)
    return mean + np.maximum(0.0, 1.0 - shrink)[None, :, None] * difference


def estimate(g, ids):
    mean = g.mean(axis=0)
    result = np.broadcast_to(mean, (N, Q)).copy()
    result[ids] += (N - 1) / (M - 1) * (g - mean)
    return result


def variance_score(g):
    return float((N - 1) / (N * (M - 1)) * np.sum(center(g) ** 2))


def probability(score):
    weight = np.sqrt(np.maximum(score, 0.0) + ETA)
    return DELTA + (1.0 - B * DELTA) * weight / weight.sum()


def make_model(seed):
    rng = np.random.default_rng(seed)
    theta = np.zeros((N, B, Q))
    for block, size in ((2, 0.85), (3, 1.2)):
        v = center(rng.normal(size=(N, Q)))
        theta[:, block] = size * v / np.sqrt(norm2(v))
    population = 0.4 + theta
    observations = population[:, None] + 0.04 * rng.normal(size=(N, 32, B, Q))
    return dict(empirical=observations.mean(axis=1), population=population, theta=theta,
                tau=np.array([0.15, 0.15, 0.25, 0.30]),
                observations_sha256=hashlib.sha256(observations.tobytes()).hexdigest())


def selected_gradient(u, model, ids, block):
    x = u[ids, block]
    return x + 5.0 * np.sin(x) - model["empirical"][ids, block]


def full_gradient(u, model):
    return u + 5.0 * np.sin(u) - model["empirical"]


def mapping(u, model):
    return norm2((u - prox_all(u - MU * full_gradient(u, model), MU, model["tau"])) / MU)


def objective(u, model):
    potential = 0.5 * np.sum(u * u, axis=(1, 2)) + 5.0 * np.sum(1 - np.cos(u), axis=(1, 2))
    return float(np.mean(potential - np.sum(model["empirical"] * u, axis=(1, 2)))
                 + model["tau"] @ radii(u))


def local_checks():
    from itertools import combinations
    rng = np.random.default_rng(87102)
    n, m = 5, 3
    g = rng.normal(size=(n, 3))
    all_scores = [(n - 1) / (n * (m - 1)) * np.sum(center(g[list(ids)]) ** 2)
                  for ids in combinations(range(n), m)]
    variance_error = abs(float(np.mean(all_scores)) - norm2(center(g)))
    model = make_model(51000)
    u = rng.normal(size=(N, B, Q))
    grad = full_gradient(u, model)
    selected_error = max(float(np.max(np.abs(selected_gradient(u, model, np.arange(N), b)
                                             - grad[:, b]))) for b in range(B))
    block_error = float(np.max(np.abs(prox_all(u, .03, model["tau"])
                         - np.stack([prox_block(u[:, b], .03, model["tau"][b]) for b in range(B)], axis=1))))
    p = probability(np.array([0.0, .1, 1.0, 100.0]))
    assert abs(p.sum() - 1) < 1e-14 and np.all(p >= DELTA)
    assert GAMMA / DELTA <= 1.0 / (9.0 * L)
    assert max(variance_error, selected_error, block_error) < 1e-12
    return dict(sample_variance_unbiasedness_error=variance_error,
                selected_gradient_consistency_error=selected_error, prox_consistency_error=block_error)


def schedule(seed):
    rng = np.random.default_rng(seed)
    pilot_ids = np.stack([rng.choice(N, M, replace=False) for _ in range(B)])
    uniforms = rng.random(BUDGET // M)
    subsets = np.stack([rng.choice(N, M, replace=False) for _ in range(BUDGET // M)])
    digest = hashlib.sha256(pilot_ids.tobytes() + uniforms.tobytes() + subsets.tobytes()).hexdigest()
    return pilot_ids, uniforms, subsets, digest


def run_method(model, method, draws):
    pilot_ids, uniforms, subsets, _ = draws
    u = np.full((N, B, Q), 3.6)
    pilot_calls = B * M if method != "uniform" else 0
    steps = (BUDGET - pilot_calls) // M
    training_seconds = 0.0
    diagnostic_seconds = 0.0
    start = time.perf_counter()
    if pilot_calls:
        score = np.array([variance_score(selected_gradient(u, model, pilot_ids[b], b)) for b in range(B)])
        p = probability(score)
    else:
        score = np.zeros(B)
        p = np.full(B, 1 / B)
    training_seconds += time.perf_counter() - start
    initial_p = p.copy()
    first_hits = {str(target): None for target in TARGETS}
    sum_mapping = 0.0
    initial_mapping = None
    ever_split = np.zeros(B, dtype=bool)
    trace = []
    p_total = np.zeros(B)
    visits = np.zeros(B, dtype=int)
    for t in range(steps + 1):
        start = time.perf_counter()
        current_mapping = mapping(u, model)
        radius = radii(u)
        split = (radius > ZERO_TOL) & SHARED
        fusion = (radius <= ZERO_TOL) & ~SHARED
        ever_split |= split
        cost = pilot_calls + M * t
        if initial_mapping is None:
            initial_mapping = current_mapping
        for target in TARGETS:
            if current_mapping <= target and first_hits[str(target)] is None:
                first_hits[str(target)] = cost
        if t % 25 == 0 or t == steps:
            trace.append(dict(updates=t, calls=cost, mapping=current_mapping,
                              budget_mapping=(pilot_calls * initial_mapping + M * sum_mapping) / cost
                              if cost else initial_mapping,
                              objective=objective(u, model), p=p.tolist(), radius=radius.tolist()))
        diagnostic_seconds += time.perf_counter() - start
        if t == steps:
            break
        sum_mapping += current_mapping
        start = time.perf_counter()
        # p is measurable BEFORE this draw; common uniforms pair the methods.
        p_total += p
        block = min(B - 1, int(np.searchsorted(np.cumsum(p), uniforms[t], side="right")))
        ids = subsets[t]
        g = selected_gradient(u, model, ids, block)
        v = estimate(g, ids)
        h = GAMMA / p[block]
        u[:, block] = prox_block(u[:, block] - h * v, h, model["tau"][block])
        visits[block] += 1
        if method == "adaptive_ema":
            score[block] = (1.0 - BETA) * score[block] + BETA * variance_score(g)
            p = probability(score)
        training_seconds += time.perf_counter() - start
        assert np.all(np.isfinite(u)) and np.all(p >= DELTA) and abs(p.sum() - 1) < 1e-12
    assert pilot_calls + M * steps == BUDGET
    return dict(pilot_calls=pilot_calls, completed_updates=steps, training_calls=BUDGET,
                update_calls=M * steps, final_mapping=current_mapping,
                update_prefix_mapping=sum_mapping / steps,
                budget_mapping=(pilot_calls * initial_mapping + M * sum_mapping) / BUDGET,
                final_objective=objective(u, model), first_hits=first_hits,
                final_shared_false_split_rate=float(np.mean(split[SHARED])),
                final_nonshared_false_fusion_rate=float(np.mean(fusion[~SHARED])),
                any_shared_false_split_rate=float(np.mean(ever_split[SHARED])),
                final_radii=radius.tolist(), initial_p=initial_p.tolist(), final_p=p.tolist(),
                average_p=(p_total / steps).tolist(), visits=visits.tolist(),
                score_bytes=int(score.nbytes) if pilot_calls else 0,
                extra_oracle_calls_for_ema=0, ema_score_updates=steps if method == "adaptive_ema" else 0,
                diagnostic_full_gradient_evaluations=steps + 1,
                diagnostic_client_block_calls=N * B * (steps + 1),
                training_seconds=training_seconds, diagnostic_seconds=diagnostic_seconds,
                trace=trace)


def mean_sd(xs):
    a = np.asarray(xs, dtype=float)
    return dict(mean=a.mean(axis=0).tolist(), sd=a.std(axis=0, ddof=1).tolist())


def summarize(runs):
    result = {}
    numeric = ("final_mapping", "update_prefix_mapping", "budget_mapping", "final_objective",
               "final_shared_false_split_rate", "final_nonshared_false_fusion_rate",
               "any_shared_false_split_rate", "training_seconds", "diagnostic_seconds", "average_p")
    for method in METHODS:
        records = [run["methods"][method] for run in runs]
        result[method] = {key: mean_sd([x[key] for x in records]) for key in numeric}
        result[method]["first_hits"] = {}
        for target in TARGETS:
            hits = [x["first_hits"][str(target)] for x in records if x["first_hits"][str(target)] is not None]
            result[method]["first_hits"][str(target)] = dict(
                reached=len(hits), total=REPEATS, conditional_median_calls=float(np.median(hits)) if hits else None,
                final_reached=sum(x["final_mapping"] <= target for x in records),
                restricted_mean_calls=float(np.mean([x["first_hits"][str(target)]
                    if x["first_hits"][str(target)] is not None else BUDGET for x in records])))
    for method in METHODS[1:]:
        result["paired_" + method + "_minus_uniform"] = {}
        for key in ("final_mapping", "budget_mapping", "final_objective"):
            differences = np.asarray([r["methods"][method][key] - r["methods"]["uniform"][key] for r in runs])
            half = 2.093024054408263 * differences.std(ddof=1) / np.sqrt(REPEATS)
            result["paired_" + method + "_minus_uniform"][key] = dict(
                mean=float(differences.mean()), ci95_lower=float(differences.mean() - half),
                ci95_upper=float(differences.mean() + half))
    return result


def build():
    checks = local_checks()
    runs = []
    for seed in range(REPEATS):
        model = make_model(51000 + seed)
        draws = schedule(71000 + seed)
        methods = {}
        # Rotate execution order to reduce systematic timing-order effects.
        for method in np.roll(np.asarray(METHODS), seed % len(METHODS)):
            methods[str(method)] = run_method(model, str(method), draws)
        error = center(model["empirical"] - model["population"])
        cm = (N - 1) / (M - 1) * np.sqrt(M / N)
        certificate = cm * np.max(np.linalg.norm(error, axis=2), axis=0)
        runs.append(dict(data_seed=51000 + seed, algorithm_seed=71000 + seed,
                         schedule_sha256=draws[-1], observations_sha256=model["observations_sha256"],
                         sharing_certificate=certificate.tolist(), methods=methods))
        print(f"Completed {seed + 1}/{REPEATS} paired seeds", flush=True)
    return dict(config=dict(N=N, B=B, Q=Q, M=M, repeats=REPEATS, budget=BUDGET,
                            delta=DELTA, beta=BETA, eta=ETA, L=L, mu=MU, gamma=GAMMA,
                            targets_squared_mapping=list(TARGETS), methods=list(METHODS),
                            model="Original separable nonconvex benchmark; amplitude 5, initial 3.6, tau (.15,.15,.25,.30)",
                            score="(N-1)/(N*(M-1))*sum_i ||g_i-gbar||^2; empirical EMA, not an upper bound",
                            probability="delta+(1-B*delta)*sqrt(score+eta)/sum sqrt(score+eta)",
                            pilot="One uniform M-subset per block at U0; paid B*M calls for both fixed and adaptive",
                            pairing="Same data, initial state, independent uniform block quantiles and uniform node subsets; distinct p produce different realized blocks",
                            oracle_unit="One exact current client/block empirical gradient (Q components)",
                            diagnostics="Full mapping evaluated at every state, separately counted and timed; never feeds training/probabilities",
                            first_hit="First observed instantaneous full squared mapping below target; includes pilot, excludes diagnostics; not a certified stopping rule",
                            timing="Single-process dense NumPy training only, includes pilot and probability updates; excludes diagnostic path and data generation; not system speed evidence",
                            adaptation="Only the selected score changes, after current sampling, for the next draw; zero extra gradient calls",
                            communication="Not measured; dense state updates do not instantiate deferred communication",
                            tuning="All settings fixed before results; no method-specific tuning or excluded seeds"),
                checks=checks, runs=runs, summary=summarize(runs))


def strip_timing(obj):
    if isinstance(obj, dict):
        return {k: strip_timing(v) for k, v in obj.items() if "seconds" not in k}
    if isinstance(obj, list):
        return [strip_timing(v) for v in obj]
    return obj


def numerical_hash(result):
    return hashlib.sha256(json.dumps(strip_timing(result), sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_outputs(result):
    (HERE / "allocation_results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    columns = ["data_seed", "algorithm_seed", "method", "training_calls", "pilot_calls", "completed_updates",
               "final_mapping", "budget_mapping", "final_objective", "final_shared_false_split_rate",
               "final_nonshared_false_fusion_rate", "training_seconds", "diagnostic_seconds"]
    with (HERE / "allocation_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for run in result["runs"]:
            for method, record in run["methods"].items():
                row = dict(data_seed=run["data_seed"], algorithm_seed=run["algorithm_seed"], method=method)
                row.update({key: record[key] for key in columns[3:]})
                writer.writerow(row)
    text = ["Block-probability ablation: 20 paired original-model data seeds, 32000 paid calls.",
            "All gradients are exact/current. Diagnostics cannot affect updates or probability scores.",
            "EMA is heuristic: no score is asserted to be a true variance upper bound.",
            "First hits are instantaneous observed crossings, not convergence guarantees or deployable free stopping.",
            json.dumps(result["summary"], indent=2), ""]
    (HERE / "allocation_summary.txt").write_text("\n".join(text), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-check", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    result = build()
    elapsed = time.perf_counter() - started
    digest = numerical_hash(result)
    replay = False
    if args.replay_check:
        again = build()
        assert numerical_hash(again) == digest, "Numerical replay mismatch"
        replay = True
    result["execution"] = dict(python=platform.python_version(), numpy=np.__version__,
                               numerical_sha256=digest, numerical_replay_passed=replay,
                               wall_seconds=elapsed, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_outputs(result)
    print(json.dumps(result["summary"], indent=2), flush=True)
    print(f"Saved results; first-run wall time {elapsed:.2f}s, numerical replay={replay}", flush=True)


if __name__ == "__main__":
    main()
