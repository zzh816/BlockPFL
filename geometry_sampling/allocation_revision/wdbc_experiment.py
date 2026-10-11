"""Real-data, coupled smooth convex example for the probability ablation.

The clients are ARTIFICIAL label-skew partitions of public UCI WDBC data, not
observed persons or clinical deployment. No true sharing labels are asserted.
The structural common-potential theorem does not apply to these local losses.
Requires NumPy and allocation_experiment.py in the same directory.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import allocation_experiment as common

HERE = Path(__file__).resolve().parent
N, M, B, D = 8, 4, 3, 31
BLOCKS = (np.arange(0, 11), np.arange(11, 21), np.arange(21, 31))
REPEATS, BUDGET = 20, 12000
DELTA, BETA, ETA, RIDGE, TAU = 0.1, 0.1, 1e-12, 0.05, 0.04
TARGETS = (0.1, 0.01, 0.001)
DATA_SHA256 = "d606af411f3e5be8a317a5a8b652b425aaf0ff38ca683d5327ffff94c3695f4a"
DATA_SOURCE = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"


def probability(score):
    weight = np.sqrt(np.maximum(score, 0) + ETA)
    return DELTA + (1 - B * DELTA) * weight / weight.sum()


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


def load_data():
    path = HERE / "wdbc.data"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == DATA_SHA256
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    x = np.asarray([[float(v) for v in row[2:]] for row in rows])
    y = np.asarray([float(row[1] == "M") for row in rows])
    assert x.shape == (569, 30) and set(y) == {0.0, 1.0}
    return x, y


def make_partition(seed):
    x, y = load_data()
    rng = np.random.default_rng(seed)
    train_ids, test_ids = [], []
    for label in (0, 1):
        ids = rng.permutation(np.flatnonzero(y == label))
        count = int(np.floor(.75 * len(ids)))
        train_ids.extend(ids[:count].tolist())
        test_ids.extend(ids[count:].tolist())
    train_ids, test_ids = np.asarray(train_ids), np.asarray(test_ids)
    mean, scale = x[train_ids].mean(axis=0), x[train_ids].std(axis=0)
    assert np.all(scale > 0)
    x = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    clients = {kind: [None] * N for kind in ("train", "test")}
    assignment = {}
    for kind, ids in (("train", train_ids), ("test", test_ids)):
        buckets = [[] for _ in range(N)]
        for label in (0, 1):
            current = rng.permutation(ids[y[ids] == label])
            # The first four clients receive 80% of the malignant pool and 20%
            # of the benign pool, split evenly within each half; vice versa.
            weight = np.array([.2] * 4 + [.8] * 4) if label == 0 else np.array([.8] * 4 + [.2] * 4)
            allocations = rng.choice(N, size=len(current), p=weight / weight.sum())
            for i in range(N):
                buckets[i].extend(current[allocations == i].tolist())
        for i, ids_i in enumerate(buckets):
            ids_i = np.asarray(ids_i, dtype=int)
            assert len(ids_i) > 0
            clients[kind][i] = (x[ids_i], y[ids_i], ids_i)
        assignment[kind] = [item[2].tolist() for item in clients[kind]]
    # Hessian <= X_i^T X_i/(4*n_i)+ridge I, including the intercept.
    smoothness = max(float(np.linalg.norm(xx, 2) ** 2 / (4 * len(yy)) + RIDGE)
                     for xx, yy, _ in clients["train"])
    clients.update(L=smoothness, mu=1 / (2 * smoothness), gamma=DELTA / (12 * smoothness),
                   train_mean=mean.tolist(), train_scale=scale.tolist(), assignment=assignment,
                   train_sizes=[len(item[1]) for item in clients["train"]],
                   test_sizes=[len(item[1]) for item in clients["test"]],
                   train_malignant_fractions=[float(item[1].mean()) for item in clients["train"]])
    return clients


def selected_gradient(u, model, ids, block):
    columns = BLOCKS[block]
    values = []
    reads = 0
    for i in ids:
        x, y, _ = model["train"][i]
        residual = sigmoid(x @ u[i]) - y
        values.append(x[:, columns].T @ residual / len(y) + RIDGE * u[i, columns])
        reads += len(y)
    return np.stack(values), reads


def full_gradient(u, model):
    result = np.empty_like(u)
    for i, (x, y, _) in enumerate(model["train"]):
        result[i] = x.T @ (sigmoid(x @ u[i]) - y) / len(y) + RIDGE * u[i]
    return result


def block_radii(u):
    return np.array([np.sqrt(common.norm2(common.center(u[:, ids]))) for ids in BLOCKS])


def diagnostic(u, model):
    g = full_gradient(u, model)
    prox = np.empty_like(u)
    for columns in BLOCKS:
        prox[:, columns] = common.prox_block(u[:, columns] - model["mu"] * g[:, columns], model["mu"], TAU)
    mapping = common.norm2((u - prox) / model["mu"])
    train_loss = np.mean([np.mean(np.logaddexp(0, x @ u[i]) - y * (x @ u[i]))
                          + .5 * RIDGE * np.sum(u[i] ** 2) for i, (x, y, _) in enumerate(model["train"])])
    return mapping, float(train_loss + TAU * block_radii(u).sum())


def test_metrics(u, model):
    losses, errors, sizes = [], [], []
    for i, (x, y, _) in enumerate(model["test"]):
        logits = x @ u[i]
        losses.append(float(np.mean(np.logaddexp(0, logits) - y * logits)))
        errors.append(float(np.mean((logits >= 0) != y)))
        sizes.append(len(y))
    return dict(equal_client_test_logloss=float(np.mean(losses)),
                equal_client_test_error=float(np.mean(errors)),
                pooled_test_logloss=float(np.average(losses, weights=sizes)),
                pooled_test_error=float(np.average(errors, weights=sizes)))


def schedule(seed):
    rng = np.random.default_rng(seed)
    pilot = np.stack([rng.choice(N, M, replace=False) for _ in range(B)])
    uniforms = rng.random(BUDGET // M)
    subsets = np.stack([rng.choice(N, M, replace=False) for _ in range(BUDGET // M)])
    return pilot, uniforms, subsets


def run_method(model, method, draws):
    pilot, uniforms, subsets = draws
    u = np.zeros((N, D))
    pilot_calls = B * M if method != "uniform" else 0
    steps = (BUDGET - pilot_calls) // M
    training_seconds = diagnostic_seconds = 0.0
    sample_reads = pilot_sample_reads = 0
    start = time.perf_counter()
    if pilot_calls:
        score = np.zeros(B)
        for b in range(B):
            g, reads = selected_gradient(u, model, pilot[b], b)
            sample_reads += reads
            score[b] = (N - 1) / (N * (M - 1)) * np.sum(common.center(g) ** 2)
        pilot_sample_reads = sample_reads
        p = probability(score)
    else:
        score, p = np.zeros(B), np.full(B, 1 / B)
    training_seconds += time.perf_counter() - start
    initial_p = p.copy()
    first_hits = {str(target): None for target in TARGETS}
    sum_mapping = 0.0
    initial_mapping = None
    trace = []
    for t in range(steps + 1):
        start = time.perf_counter()
        mapping, objective = diagnostic(u, model)
        if initial_mapping is None:
            initial_mapping = mapping
        cost = pilot_calls + M * t
        for target in TARGETS:
            if mapping <= target and first_hits[str(target)] is None:
                first_hits[str(target)] = dict(calls=cost, sample_reads=sample_reads)
        if t % 25 == 0 or t == steps:
            trace.append(dict(calls=cost, sample_reads=sample_reads, mapping=mapping,
                              budget_mapping=(pilot_calls * initial_mapping + M * sum_mapping) / cost if cost else mapping,
                              objective=objective, p=p.tolist()))
        diagnostic_seconds += time.perf_counter() - start
        if t == steps:
            break
        sum_mapping += mapping
        start = time.perf_counter()
        block = min(B - 1, int(np.searchsorted(np.cumsum(p), uniforms[t], side="right")))
        ids = subsets[t]
        g, reads = selected_gradient(u, model, ids, block)
        sample_reads += reads
        gmean = g.mean(axis=0)
        v = np.broadcast_to(gmean, (N, len(BLOCKS[block]))).copy()
        v[ids] += (N - 1) / (M - 1) * (g - gmean)
        h = model["gamma"] / p[block]
        columns = BLOCKS[block]
        u[:, columns] = common.prox_block(u[:, columns] - h * v, h, TAU)
        if method == "adaptive_ema":
            value = (N - 1) / (N * (M - 1)) * np.sum(common.center(g) ** 2)
            score[block] = (1 - BETA) * score[block] + BETA * value
            p = probability(score)
        training_seconds += time.perf_counter() - start
        assert np.all(p >= DELTA) and np.all(np.isfinite(u))
    result = dict(training_calls=BUDGET, pilot_calls=pilot_calls, pilot_sample_reads=pilot_sample_reads,
                  sample_reads=sample_reads, completed_updates=steps,
                  final_mapping=mapping, final_objective=objective,
                  budget_mapping=(pilot_calls * initial_mapping + M * sum_mapping) / BUDGET,
                  update_prefix_mapping=sum_mapping / steps, first_hits=first_hits,
                  training_seconds=training_seconds, diagnostic_seconds=diagnostic_seconds,
                  initial_p=initial_p.tolist(), final_p=p.tolist(), trace=trace,
                  diagnostic_full_gradient_evaluations=steps + 1,
                  diagnostic_client_block_calls=N * B * (steps + 1),
                  diagnostic_gradient_sample_reads=sum(model["train_sizes"]) * (steps + 1),
                  extra_oracle_calls_for_ema=0, final_radii=block_radii(u).tolist())
    result.update(test_metrics(u, model))
    return result


def checks(model):
    rng = np.random.default_rng(8880)
    u = rng.normal(size=(N, D)) * .1
    g = full_gradient(u, model)
    consistency = max(float(np.max(np.abs(selected_gradient(u, model, np.arange(N), b)[0]
                                          - g[:, BLOCKS[b]]))) for b in range(B))
    x, y, _ = model["train"][0]
    def local_loss(z):
        return float(np.mean(np.logaddexp(0, x @ z) - y * (x @ z)) + .5 * RIDGE * np.sum(z * z))
    eps = 1e-6
    plus, minus = u[0].copy(), u[0].copy()
    plus[7] += eps
    minus[7] -= eps
    fd_error = abs((local_loss(plus) - local_loss(minus)) / (2 * eps) - g[0, 7])
    assert consistency < 1e-12 and fd_error < 1e-7
    train = {j for ids in model["assignment"]["train"] for j in ids}
    test = {j for ids in model["assignment"]["test"] for j in ids}
    assert train.isdisjoint(test) and len(train | test) == 569
    return dict(selected_full_gradient_error=consistency, finite_difference_error=fd_error,
                train_test_disjoint=True)


def main():
    runs = []
    verification = checks(make_partition(81000))
    started = time.perf_counter()
    for j in range(REPEATS):
        model = make_partition(81000 + j)
        draws = schedule(91000 + j)
        methods = {str(method): run_method(model, str(method), draws)
                   for method in np.roll(np.asarray(common.METHODS), j % 3)}
        runs.append(dict(split_seed=81000 + j, algorithm_seed=91000 + j,
                         model={key: model[key] for key in ("L", "mu", "gamma", "train_mean", "train_scale", "assignment",
                                                           "train_sizes", "test_sizes", "train_malignant_fractions")},
                         methods=methods))
        print(f"WDBC completed {j + 1}/{REPEATS} paired seeds", flush=True)
    summary = {}
    keys = ("final_mapping", "budget_mapping", "final_objective", "sample_reads", "training_seconds",
            "diagnostic_seconds", "equal_client_test_logloss", "equal_client_test_error", "pooled_test_logloss", "pooled_test_error")
    for method in common.METHODS:
        summary[method] = {key: common.mean_sd([r["methods"][method][key] for r in runs]) for key in keys}
        summary[method]["first_hits"] = {}
        for target in TARGETS:
            values = [r["methods"][method]["first_hits"][str(target)] for r in runs]
            hits = [v for v in values if v is not None]
            summary[method]["first_hits"][str(target)] = dict(reached=len(hits), total=REPEATS,
                conditional_median_calls=float(np.median([v["calls"] for v in hits])) if hits else None,
                conditional_median_sample_reads=float(np.median([v["sample_reads"] for v in hits])) if hits else None,
                final_reached=sum(r["methods"][method]["final_mapping"] <= target for r in runs))
    for method in common.METHODS[1:]:
        summary["paired_" + method + "_minus_uniform"] = {}
        for key in ("final_mapping", "budget_mapping", "equal_client_test_logloss", "equal_client_test_error"):
            ds = np.array([r["methods"][method][key] - r["methods"]["uniform"][key] for r in runs])
            half = 2.093024054408263 * ds.std(ddof=1) / np.sqrt(REPEATS)
            summary["paired_" + method + "_minus_uniform"][key] = dict(mean=float(ds.mean()),
                ci95_lower=float(ds.mean() - half), ci95_upper=float(ds.mean() + half))
    result = dict(config=dict(N=N, M=M, B=B, D=D, block_dimensions=[11, 10, 10], budget=BUDGET,
        repeats=REPEATS, delta=DELTA, beta=BETA, eta=ETA, ridge=RIDGE, tau=TAU,
        targets_squared_mapping=list(TARGETS), data_url=DATA_SOURCE, data_sha256=DATA_SHA256,
        dataset_doi="10.24432/C5DW2B", dataset_license="CC BY 4.0",
        reference="Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository.",
        objective="Equal client average of logistic empirical losses plus ridge .05/2||u_i||^2, plus .04 sum_b||DU_b||_N",
        preprocessing="75% classwise train split; mean/std fitted only on pooled training data; intercept added; every row retained in exactly one client train/test set",
        personalization="Artificial label-skew client partition: each of first four clients has class-conditional weight .8 for malignant and .2 for benign; reversed in last four",
        general_theory_only="Local logistic objectives are coupled across feature blocks; neither common-potential structural theorem nor true sharing labels are asserted",
        oracle_cost="Exact client/block empirical gradient. Counts unequal sample costs separately: one read of every local training example per queried block; each forward logit uses all 31 coordinates",
        diagnostics="Full deterministic mapping every iterate, separately counted/timed and never available to training; test metrics evaluated at final output only",
        timing="Dense single-process NumPy including pilot and EMA arithmetic, excluding diagnostics/data preprocessing; no communication or system speed claim",
        fixed_parameters="One common gamma=delta/(12L) per dataset; no method-specific tuning; both nonuniform strategies pay 12 pilot gradient calls",
        output="First observed instantaneous mapping crossings, not proof of deployable stopping; equal-client held-out loss/error and separately sample-weighted held-out metrics"),
        verification=verification, runs=runs, summary=summary,
        execution=dict(python=platform.python_version(), numpy=np.__version__, wall_seconds=time.perf_counter() - started,
                       source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    result["execution"]["numerical_sha256"] = common.numerical_hash({k: v for k, v in result.items() if k != "execution"})
    (HERE / "wdbc_results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (HERE / "wdbc_summary.txt").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    columns = ["split_seed", "algorithm_seed", "method"] + list(keys)
    with (HERE / "wdbc_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for run in runs:
            for method, record in run["methods"].items():
                row = {key: run[key] for key in columns[:2]}
                row["method"] = method
                row.update({key: record[key] for key in keys})
                writer.writerow(row)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
