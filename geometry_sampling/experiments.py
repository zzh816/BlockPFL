"""Paired current-gradient experiments for the 1007 geometric block method.

No stochastic gradient, stale gradient, local multistep update, or history
correction is used. Full gradients are evaluated only by the diagnostic path.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
METHODS = ("geometric", "inverse_inclusion")
REPEATS = 20
ZERO_TOL = 1e-10


def center(x):
    return x - x.mean(axis=0)


def norm2(x):
    return float(np.mean(np.sum(x * x, axis=1)))


def prox_block(y, h, tau):
    w = y.mean()
    d = y - w
    r = float(np.sqrt(np.mean(d * d)))
    alpha = max(0.0, 1.0 - h * tau / r) if r else 0.0
    return w + alpha * d


def estimate(sampled_gradients, ids, n, method):
    """Input contains exactly M current client/block gradients."""
    m = len(ids)
    if method == "geometric":
        mean = sampled_gradients.mean()
        v = np.full(n, mean)
        v[ids] += (n - 1) / (m - 1) * (sampled_gradients - mean)
    else:
        v = np.zeros(n)
        v[ids] = n / m * sampled_gradients
    return v


@dataclass
class Model:
    name: str
    kind: str
    empirical: np.ndarray
    population: np.ndarray
    theta: np.ndarray
    tau: np.ndarray
    L: float
    initial: float
    coupling: float = 0.0

    @property
    def n(self):
        return self.empirical.shape[0]

    @property
    def b(self):
        return self.empirical.shape[1]

    def sampled_gradient(self, u, ids, k):
        # Only selected clients and the selected derivative are computed here.
        v = u[ids, k] - self.empirical[ids, k]
        if self.kind == "nonconvex":
            v = v + 10.0 * np.sin(u[ids, k])
        if self.kind == "coupled":
            v = v + self.coupling * u[ids, 1 - k]
        return v

    def full_gradient(self, u):
        v = u - self.empirical
        if self.kind == "nonconvex":
            v = v + 10.0 * np.sin(u)
        if self.kind == "coupled":
            v = v + self.coupling * u[:, ::-1]
        return v

    def potential(self, u):
        h = 0.5 * np.sum(u * u, axis=1)
        if self.kind == "nonconvex":
            h += 10.0 * np.sum(1.0 - np.cos(u), axis=1)
        if self.kind == "coupled":
            h += self.coupling * u[:, 0] * u[:, 1]
        return h

    def objective(self, u):
        r = np.sqrt(np.mean(center(u) ** 2, axis=0))
        return float(np.mean(self.potential(u) - np.sum(self.empirical * u, axis=1))
                     + self.tau @ r)

    def paired_population_gain(self, u):
        w = np.broadcast_to(u.mean(axis=0), u.shape)
        # Risk(shared output) - risk(personalized output), on the same output.
        return float(np.mean(self.potential(w) - self.potential(u)
                     + np.sum(self.population * (u - w), axis=1)))

    def metadata(self):
        return dict(name=self.name, kind=self.kind, N=self.n, B=self.b, L=self.L,
                    initial=self.initial, coupling=self.coupling,
                    empirical_mean=self.empirical.tolist(),
                    population_mean=self.population.tolist(), theta=self.theta.tolist(),
                    tau=self.tau.tolist())


def full_mapping(u, model):
    mu = 1.0 / (2.0 * model.L)
    g = model.full_gradient(u)
    v = np.column_stack([prox_block(u[:, k] - mu * g[:, k], mu, model.tau[k])
                         for k in range(model.b)])
    return norm2((u - v) / mu), g


def exact_variance_checks():
    n = 8
    signs = np.array([-1.0, 1.0] * 4)
    rows = []
    for name, mean, spread in (("common", 2.0, 0.0),
                               ("large_mean_small_heterogeneity", 2.0, 0.1),
                               ("zero_mean_large_heterogeneity", 0.0, 0.8)):
        g = mean + spread * signs
        z2 = float(np.mean(center(g) ** 2))
        for m in (2, 4, 8):
            errors = {method: [] for method in METHODS}
            means = {method: [] for method in METHODS}
            ds = {method: [] for method in METHODS}
            for subset in combinations(range(n), m):
                ids = np.array(subset)
                for method in METHODS:
                    err = estimate(g[ids], ids, n, method) - g
                    errors[method].append(float(np.mean(err * err)))
                    means[method].append(float(err.mean() ** 2))
                    ds[method].append(float(np.mean(center(err) ** 2)))
            am = (n - m) / (m * (n - 1))
            km = (n - m) / (m - 1)
            formulas = {"geometric": (am + km) * z2,
                        "inverse_inclusion": (n - m) / m * float(np.mean(g * g))}
            row = dict(regime=name, N=n, M=m, gradient_mean=mean,
                       gradient_disagreement_squared=z2, gradients=g.tolist(),
                       advantage_margin=mean ** 2 - (1 / (m - 1) + 1 / (n - 1)) * z2)
            for method in METHODS:
                row[method] = dict(total=float(np.mean(errors[method])),
                                   mean_channel=float(np.mean(means[method])),
                                   centered_channel=float(np.mean(ds[method])),
                                   formula=formulas[method])
                assert abs(row[method]["total"] - formulas[method]) < 2e-12
            rows.append(row)
    return rows


def run_pair(model, m, T, gamma, p, algorithm_seed, data_seed=None,
             conditional_coupled_bounds=None):
    n, b = model.n, model.b
    assert 2 <= m <= n and len(p) == b and np.all(p > 0)
    assert abs(float(p.sum()) - 1.0) < 1e-12
    rr = np.random.default_rng(algorithm_seed)
    # Generate independently of either trajectory; both methods receive these draws.
    blocks = rr.choice(b, size=T, p=p)
    subsets = np.array([rr.choice(n, m, replace=False) for _ in range(T)])
    schedule_hash = hashlib.sha256(blocks.tobytes() + subsets.tobytes()).hexdigest()
    us = {method: np.full((n, b), model.initial) for method in METHODS}
    sums = {method: dict(mapping=0.0, variance=0.0, advantage=0,
                       split=np.zeros(b), fusion=np.zeros(b), ever=np.zeros(b, dtype=bool),
                       max_deviation=np.zeros(b), max_radius=np.zeros(b), max_abs=0.0) for method in METHODS}
    checkpoints = sorted(set([0, T] + [max(1, int(T * x))
                                      for x in (0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8)]))
    traces = {method: [] for method in METHODS}
    shared = np.sqrt(np.mean(model.theta ** 2, axis=0)) <= ZERO_TOL
    km = (n - m) / (m - 1)
    am = (n - m) / (m * (n - 1))
    coupled = {method: dict(region_violations=0, deviation_violations=0, radius_violations=0,
                           score_violations_at_shared=0, inspected_shared_updates=0)
               for method in METHODS}
    for t in range(T + 1):
        for method in METHODS:
            u = us[method]
            mapping, g = full_mapping(u, model)
            rad = np.sqrt(np.mean(center(u) ** 2, axis=0))
            s = sums[method]
            s["max_deviation"] = np.maximum(s["max_deviation"],
                                             np.max(np.abs(center(u)), axis=0))
            s["max_radius"] = np.maximum(s["max_radius"], rad)
            s["max_abs"] = max(s["max_abs"], float(np.max(np.abs(u))))
            # Include the final state in the event "any split during the run".
            s["ever"] |= (rad > ZERO_TOL) & shared
            if t in checkpoints:
                traces[method].append(dict(updates=t, training_oracles=m * t,
                                           instantaneous_mapping=mapping,
                                           running_mapping=s["mapping"] / t if t else mapping,
                                           objective=model.objective(u), radii=rad.tolist()))
            if t == T:
                continue
            s["mapping"] += mapping
            s["split"] += (rad > ZERO_TOL) & shared
            s["fusion"] += (rad <= ZERO_TOL) & ~shared
            z2 = np.mean(center(g) ** 2, axis=0)
            mg2 = g.mean(axis=0) ** 2
            s["advantage"] += int(float(p @ (mg2 - (1 / (m - 1) + 1 / (n - 1)) * z2)) >= 0)
            var = (am + km) * z2 if method == "geometric" else (n - m) / m * (z2 + mg2)
            s["variance"] += float(p @ var)
        if t == T:
            break
        k, ids = int(blocks[t]), subsets[t]
        h = gamma / p[k]
        for method in METHODS:
            u = us[method]
            if conditional_coupled_bounds is not None:
                bounds = conditional_coupled_bounds
                c = coupled[method]
                if bounds.get("box_radius") is not None:
                    c["region_violations"] += int(np.max(np.abs(u)) > bounds["box_radius"] + 1e-12)
                dev = float(np.max(np.abs(center(u)[:, 1])))
                c["deviation_violations"] += int(dev > bounds["rho_other"] + 1e-12)
                c["radius_violations"] += int(np.sqrt(np.mean(center(u[:, 1]) ** 2))
                                              > bounds["radius_other_bound"] + 1e-12)
                if k == 0 and np.sqrt(np.mean(center(u[:, 0]) ** 2)) <= ZERO_TOL:
                    c["inspected_shared_updates"] += 1
                    score = (n - 1) / (m - 1) * np.sqrt(m / n) * (
                        bounds["error_max_block0"] + model.coupling * bounds["rho_other"])
                    c["score_violations_at_shared"] += int(score > model.tau[0] + 1e-12)
            # Evaluate each method's current state, never reuse the other gradient.
            sampled = model.sampled_gradient(u, ids, k)
            v = estimate(sampled, ids, n, method)
            u[:, k] = prox_block(u[:, k] - h * v, h, model.tau[k])
            assert np.all(np.isfinite(u))
    records = {}
    for method in METHODS:
        u, s = us[method], sums[method]
        final_map, _ = full_mapping(u, model)
        rad = np.sqrt(np.mean(center(u) ** 2, axis=0))
        records[method] = dict(mean_squared_full_mapping=s["mapping"] / T,
            final_squared_full_mapping=final_map, final_objective=model.objective(u),
            paired_population_gain=model.paired_population_gain(u),
            mean_conditional_estimator_variance=s["variance"] / T,
            own_state_advantage_fraction=s["advantage"] / T,
            any_false_split=s["ever"].tolist(),
            random_prefix_false_split_fraction=(s["split"] / T).tolist(),
            random_prefix_false_fusion_fraction=(s["fusion"] / T).tolist(),
            final_false_split=((rad > ZERO_TOL) & shared).tolist(),
            final_false_fusion=((rad <= ZERO_TOL) & ~shared).tolist(),
            final_radii=rad.tolist(), max_coordinate_abs=s["max_abs"],
            max_client_deviation=s["max_deviation"].tolist(), final_state=u.tolist(),
            max_block_radius=s["max_radius"].tolist(),
            trace=traces[method], training_client_block_oracles=m * T,
            # T+1 diagnostics in loop, and one explicit final recomputation.
            diagnostic_client_block_oracles=n * b * (T + 2),
            conditional_coupled_checks=coupled[method])
        if conditional_coupled_bounds is not None and method == "geometric":
            assert not s["ever"][0]
            assert s["max_radius"][1] <= conditional_coupled_bounds["radius_other_bound"] + 1e-12
    if m == n:
        assert np.max(np.abs(us[METHODS[0]] - us[METHODS[1]])) < 2e-12
    return dict(data_seed=data_seed, algorithm_seed=algorithm_seed,
                participation_sequence_sha256=schedule_hash, methods=records)


def mean_sd(values):
    x = np.asarray(values, dtype=float)
    return dict(mean=np.mean(x, axis=0).tolist(), sd=np.std(x, axis=0, ddof=1).tolist())


def summarize(runs):
    numeric = ("mean_squared_full_mapping", "final_squared_full_mapping", "final_objective",
               "paired_population_gain", "mean_conditional_estimator_variance",
               "own_state_advantage_fraction", "random_prefix_false_split_fraction",
               "random_prefix_false_fusion_fraction", "final_radii", "max_client_deviation")
    out = {}
    for method in METHODS:
        out[method] = {key: mean_sd([r["methods"][method][key] for r in runs]) for key in numeric}
        for key in ("any_false_split", "final_false_split", "final_false_fusion"):
            out[method][key] = mean_sd([r["methods"][method][key] for r in runs])
        out[method]["trace"] = []
        for j, point in enumerate(runs[0]["methods"][method]["trace"]):
            q = dict(updates=point["updates"], training_oracles=point["training_oracles"])
            for key in ("instantaneous_mapping", "running_mapping", "objective", "radii"):
                q[key] = mean_sd([r["methods"][method]["trace"][j][key] for r in runs])
            out[method]["trace"].append(q)
    out["paired_geometric_minus_inverse_inclusion"] = {
        key: mean_sd([np.asarray(r["methods"]["geometric"][key])
                       - np.asarray(r["methods"]["inverse_inclusion"][key]) for r in runs])
        for key in numeric}
    return out


def make_group(name, models, m, T, gamma, p, seed_base, data_seeds=None, bounds=None):
    print(f"Running {name}: M={m}, T={T}, repetitions={len(models)}", flush=True)
    runs = [run_pair(model, m, T, gamma, p, seed_base + j,
                     None if data_seeds is None else data_seeds[j],
                     None if bounds is None else bounds[j]) for j, model in enumerate(models)]
    return dict(name=name, M=m, T=T, gamma=gamma, block_probabilities=p.tolist(),
                selected_block_steps=(gamma / p).tolist(), repetitions=len(models),
                models=[x.metadata() for x in models], runs=runs, summary=summarize(runs))


def build_results(legacy):
    n = 8
    signs = np.array([-1.0, 1.0] * 4)[:, None]
    groups = []
    for name, mean, signal in (("common_gradient", 2.0, 0.0),
                                ("large_mean_small_heterogeneity", 2.0, 0.1),
                                ("zero_mean_large_heterogeneity", 0.0, 0.8)):
        theta = signal * signs
        pop = mean + theta
        model = Model(name, "quadratic", pop, pop, theta, np.array([0.12]), 1.0, 0.0)
        groups.append(make_group(name, [model] * REPEATS, 2, 1600, 0.03,
                                 np.array([1.0]), 20000))
    old = legacy["instance"]
    original = Model("nonconvex_common_tilt", "nonconvex", np.asarray(old["empirical_mean"]),
                     np.asarray(old["population_mean"]), np.asarray(old["theta"]),
                     np.asarray(old["tau"]), old["L"], 0.4)
    # Across M use the same total training budget; within every pair all settings coincide.
    for m in (2, 4, 8):
        groups.append(make_group(f"nonconvex_M{m}", [original] * REPEATS, m, 6400 // m,
                                old["stepmax"], np.full(4, 0.25), 21000))
    tau_sweep = []
    for tau in (0.0, 0.05, 0.14, 0.25, 0.6, 1.0):
        theta = np.hstack((np.zeros_like(signs), 0.5 * signs))
        pop = np.array([1.5, 0.0]) + theta
        # Realized errors fixed at +/-0.04; repeat only the update randomness.
        empirical = pop + np.hstack((0.04 * signs, 0.04 * signs))
        model = Model("quadratic_threshold", "quadratic", empirical, pop, theta,
                      np.full(2, tau), 1.0, 0.0)
        tau_sweep.append(make_group(f"tau_{tau:g}", [model] * REPEATS, 2, 1200,
                                   0.015 / 2, np.full(2, 0.5), 22000))
    noise_sweep = []
    for noise in (0.0, 0.12, 0.4):
        for m in (2, 4, 8):
            models, seeds = [], []
            for j in range(REPEATS):
                seed = 31000 + j
                data = np.random.default_rng(seed).normal(size=(n, 32, 2))
                theta = np.hstack((np.zeros_like(signs), 0.5 * signs))
                pop = np.array([1.5, 0.0]) + theta
                empirical = pop + noise * data.mean(axis=1)
                models.append(Model("quadratic_noise", "quadratic", empirical, pop, theta,
                                    np.full(2, 0.25), 1.0, 0.0))
                seeds.append(seed)
            noise_sweep.append(make_group(f"noise_{noise:g}_M{m}", models, m, 2400 // m,
                                         0.015 / 2, np.full(2, 0.5), 23000, seeds))
    # All-path coupled certificate: R=c_M*A_y^2/(2*tau_y), h*c_M<=1.
    # Consensus in block x is a regularization certificate, not a population
    # optimum sharing label; Q^{-1} mixes the y signal into the x optimum.
    theta = np.hstack((np.zeros_like(signs), 0.5 * signs))
    pop = np.array([1.5, 0.0]) + theta
    empirical = pop + np.hstack((0.02 * signs, np.zeros_like(signs)))
    c_m, C_m, A_y, E_x, tau_y = 7.0, 3.5, 0.5, 0.02, 0.25
    R = c_m * A_y ** 2 / (2.0 * tau_y)
    rho_other = np.sqrt(n) * R
    score = C_m * (0.2 * rho_other + E_x)
    coupled = Model("coupled_quadratic", "coupled", empirical, pop, theta,
                    np.array([1.01 * score, tau_y]), 1.2, 0.0, 0.2)
    bounds = dict(box_radius=None, rho_other=rho_other, error_max_block0=E_x,
                  A_y=A_y, radius_other_bound=R, threshold_bound=score,
                  selected_step_times_a_y_c_m=0.015 * c_m)
    coupled_group = make_group("coupled_all_path_certificate", [coupled] * REPEATS,
                                2, 1600, 0.015 / 2, np.full(2, 0.5), 24000,
                                bounds=[bounds] * REPEATS)
    return dict(schema_version=1, repeats=REPEATS, methods=list(METHODS),
                zero_radius_tolerance=ZERO_TOL,
                protocol=dict(exact_current_gradients=True, common_participation_draws=True,
                    full_state_prox=True, no_history_or_stale_gradients=True,
                    output="uniform random prefix; exact average of pre-update diagnostics",
                    data_randomness="Original data from legacy seed 20261006; noise sweep seeds 31000..31019",
                    algorithm_randomness="Independent schedule seeds 20000..24019 as stored per group",
                    error_bars="Sample standard deviation across 20 runs, not a confidence interval",
                    diagnostics="Separate full gradients for deterministic mapping, reused for conditional variance",
                    training_budget="Exactly M*T client-block gradients per method; preprocessing and diagnostics excluded",
                    raw_sample_reads="Quadratic/common-tilt empirical means are sufficient statistics; subsequent reads zero"),
                original_dataset=old["data"], original_preprocessing_observations=legacy["preprocessing_scalar_observations"],
                legacy_checks=dict(enumeration={key:max(row[key] for row in legacy['enumeration'])
                    for key in ('unbiasedness_error','mean_variance_error','disagreement_variance_error','advantage_gap_error')},
                    lazy=legacy['lazy'], structure=legacy['structure_checks']),
                variance_enumeration=exact_variance_checks(), main_groups=groups,
                tau_sweep=tau_sweep, noise_participation_sweep=noise_sweep,
                coupled_group=coupled_group, coupled_public_bounds=bounds,
                coupled_scope="Public all-path radius/consensus certificate; x has zero direct tilt, not necessarily shared population minimizers")


def generate(reference_path=HERE / "fixtures/mechanism_reference.json", output=HERE):
    """Regenerate the numerical archive from the fixed reference fixture."""
    result = build_results(json.loads(reference_path.read_text(encoding="utf-8")))
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-data", type=Path, default=HERE / "fixtures/mechanism_reference.json")
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    generate(args.reference_data, args.output)
