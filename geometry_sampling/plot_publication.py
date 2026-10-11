"""Create publication figures from archived per-run experiment diagnostics.

No simulations are run and no numeric source files are modified. Reproduce:
    python geometry_sampling/plot_publication.py

The continuous-data intervals are nominal pointwise 95% Student-t confidence
intervals for the run mean, not standard deviations, simultaneous bands,
or theoretical estimator variances. Bernoulli run events use Wilson 95%
intervals. The 20 runs remain paired between methods; each displayed
interval describes a method mean, not a paired method difference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "figures" / ".matplotlib-cache"))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, LogLocator, LogFormatterMathtext, NullFormatter

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "figures"
T_CRIT_19 = 2.093024054408263
NORMAL_975 = 1.959963984540054
MM = 1 / 25.4
STYLES = {
    "geometric": {"label": "Geometric", "color": "#0072B2", "ls": "-", "marker": "o"},
    "inverse_inclusion": {"label": "Inverse inclusion", "color": "#D55E00", "ls": "--", "marker": "s"},
    "client_gradient_table": {"label": "Gradient table", "color": "#8E44AD", "ls": "-.", "marker": "^"},
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure_fonts():
    font = "Arial" if any(f.name == "Arial" for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [font],
        "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 9,
        "xtick.labelsize": 7.2, "ytick.labelsize": 7.2, "legend.fontsize": 7.2,
        "mathtext.fontset": "dejavusans", "axes.linewidth": .65,
        "lines.linewidth": 1.2, "lines.markersize": 3.1,
        "xtick.major.width": .6, "ytick.major.width": .6,
        "xtick.major.size": 2.4, "ytick.major.size": 2.4,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "svg.hashsalt": "BlockPFL-publication-20261008",
        "pdf.compression": 9, "figure.dpi": 150, "savefig.dpi": 600,
    })
    return font


def mean_ci(values):
    a = np.asarray(values, dtype=float)
    if a.shape[0] != 20:
        raise ValueError(f"The archived protocol requires 20 runs; found {a.shape[0]}.")
    if not np.all(np.isfinite(a)):
        raise ValueError("Nonfinite run diagnostic.")
    mean = np.mean(a, axis=0)
    half = T_CRIT_19 * np.std(a, axis=0, ddof=1) / np.sqrt(a.shape[0])
    return mean, mean - half, mean + half


def wilson_ci(values):
    a = np.asarray(values, dtype=float)
    if a.shape[0] != 20 or not np.all((a == 0) | (a == 1)):
        raise ValueError("Wilson intervals require the 20 binary run indicators.")
    n = a.shape[0]
    p = np.mean(a, axis=0)
    z2 = NORMAL_975 ** 2
    den = 1 + z2 / n
    center = (p + z2 / (2 * n)) / den
    half = NORMAL_975 * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / den
    return p, np.minimum(p, np.maximum(0, center - half)), np.maximum(p, np.minimum(1, center + half))


def method_line(ax, x, mean, lo, hi, method, band=True):
    style = STYLES[method]
    opts = dict(color=style["color"], linestyle=style["ls"], marker=style["marker"],
                markerfacecolor="white", markeredgewidth=.85, zorder=3)
    if band:
        if ax.get_yscale() == "log" and np.any(lo <= 0):
            raise ValueError("A pointwise confidence interval reaches zero on a log axis.")
        ax.fill_between(x, lo, hi, color=style["color"], alpha=.16, linewidth=0, zorder=1)
        if len(x) > 12:
            log_x = np.log(np.asarray(x, dtype=float))
            opts["markevery"] = np.unique([np.argmin(np.abs(log_x - v))
                                          for v in np.linspace(log_x[0], log_x[-1], 8)]).tolist()
        ax.plot(x, mean, **opts)
    else:
        ax.errorbar(x, mean, yerr=np.stack([mean - lo, hi - mean]),
                    capsize=1.7, elinewidth=.8, capthick=.7, **opts)


def panel_title(ax, letter, title):
    ax.set_title(f"({letter}) {title}", loc="left", pad=4, fontweight="normal")


def stationarity_panel(ax, group, title, letter, methods, budget_weighted=False):
    panel_title(ax, letter, title)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1,), numticks=4))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10, labelOnlyBase=True))
    ax.set_xlabel("Gradient calls")
    ax.set_ylabel("Squared mapping")
    key = "budget_running_mapping" if budget_weighted else "running_mapping"
    cost_key = "training_client_block_oracles" if budget_weighted else "training_oracles"
    for method in methods:
        traces = [run["methods"][method]["trace"] for run in group["runs"]]
        costs = np.asarray([p[cost_key] for p in traces[0]])
        if any([p[cost_key] for p in trace] != costs.tolist() for trace in traces[1:]):
            raise ValueError("Run checkpoints must agree before estimating mean intervals.")
        values = np.asarray([[p[key] for p in trace] for trace in traces])
        keep = costs > 0
        mean, lo, hi = mean_ci(values[:, keep])
        method_line(ax, costs[keep], mean, lo, hi, method)
    ax.margins(x=.03, y=.10)
    ticks = [200, 1000, 5000] if max(costs) >= 5000 else [100, 1000]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([str(v) for v in ticks]))
    if budget_weighted and group["name"] == "common_potential_nonconvex":
        ax.yaxis.set_major_locator(FixedLocator([30, 100, 300]))
        ax.yaxis.set_major_formatter(FixedFormatter(["30", "100", "300"]))
    elif budget_weighted:
        ax.yaxis.set_major_locator(FixedLocator([200, 500, 1000]))
        ax.yaxis.set_major_formatter(FixedFormatter(["200", "500", "1000"]))


def figure_legend(fig, axes, methods, y=.984):
    handles = [Line2D([], [], color=STYLES[m]["color"], ls=STYLES[m]["ls"],
                      marker=STYLES[m]["marker"], mfc="white", mew=.85,
                      label=STYLES[m]["label"]) for m in methods]
    legend = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, y),
                        ncol=len(methods), frameon=False, handlelength=2.6,
                        columnspacing=2.4, borderaxespad=0)
    fig.canvas.draw()
    lb = legend.get_window_extent(fig.canvas.get_renderer())
    for ax in np.asarray(axes).ravel():
        if lb.overlaps(ax.bbox):
            raise RuntimeError("The global legend overlaps a data panel.")
    return legend


def validate_layout(fig, axes, legend):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    objects = [(f"panel {i + 1}", ax.get_tightbbox(renderer))
               for i, ax in enumerate(np.asarray(axes).ravel())]
    objects.append(("global legend", legend.get_window_extent(renderer)))
    for label, bbox in objects:
        if bbox.x0 < bounds.x0 - .5 or bbox.y0 < bounds.y0 - .5 or bbox.x1 > bounds.x1 + .5 or bbox.y1 > bounds.y1 + .5:
            raise RuntimeError(f"Content clipping in {label}: {bbox.bounds}, canvas={bounds.bounds}.")
    return {"canvas_pixels_at_figure_dpi": list(bounds.bounds),
            "legend_bounds_pixels": list(objects[-1][1].bounds),
            "panel_tight_bounds_pixels": [list(b.bounds) for _, b in objects[:-1]],
            "legend_overlaps_data": False, "content_clipped": False}


def save_figure(fig, axes, legend, name, out, title, caption):
    layout = validate_layout(fig, axes, legend)
    pdf_metadata = {"Title": title, "Subject": caption,
                    "Creator": "BlockPFL publication plotting script",
                    "CreationDate": None, "ModDate": None}
    files = {}
    for ext in ("pdf", "svg", "png"):
        path = out / f"{name}.{ext}"
        metadata = pdf_metadata if ext == "pdf" else ({"Title": title, "Description": caption, "Date": None} if ext == "svg" else {"Title": title, "Description": caption})
        fig.savefig(path, format=ext, dpi=600, metadata=metadata)
        files[ext] = {"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
    size = (np.asarray(fig.get_size_inches()) * 25.4).tolist()
    plt.close(fig)
    return {"title": title, "caption": caption, "size_mm": size, "files": files, "layout": layout}


def mechanisms(result, out, width_mm, height_mm):
    methods = ("geometric", "inverse_inclusion")
    fig, axes = plt.subplots(2, 3, figsize=(width_mm * MM, height_mm * MM))
    fig.subplots_adjust(left=.095, right=.965, bottom=.13, top=.84, wspace=.62, hspace=.95)
    groups = {g["name"]: g for g in result["main_groups"]}
    stationarity_panel(axes[0, 0], groups["large_mean_small_heterogeneity"], "Large mean", "a", methods)
    stationarity_panel(axes[0, 1], groups["zero_mean_large_heterogeneity"], "Zero mean", "b", methods)
    stationarity_panel(axes[0, 2], groups["nonconvex_M2"], "Nonconvex", "c", methods)
    tau_groups = sorted(result["tau_sweep"], key=lambda g: g["models"][0]["tau"][0])
    taus = np.asarray([g["models"][0]["tau"][0] for g in tau_groups])
    for ax, field, block, letter, title in (
            (axes[1, 0], "any_false_split", 0, "d", "Any false split"),
            (axes[1, 1], "final_false_fusion", 1, "e", "Final fusion")):
        panel_title(ax, letter, title)
        for method in methods:
            values = np.asarray([
                [float(g["runs"][j]["methods"][method][field][block]) for g in tau_groups]
                for j in range(20)])
            mean, lo, hi = wilson_ci(values)
            method_line(ax, taus, mean, lo, hi, method, band=False)
        ax.axvline(.14, color=".55", lw=.75, ls=":", zorder=0)
        ax.set_xlabel("Threshold $\\tau$")
        ax.set_ylabel("Run probability")
        ax.set_ylim(-.04, 1.08)
        ax.set_xticks([0, .25, .6, 1])
        ax.set_yticks([0, .5, 1])
        ax.margins(x=.045)
    ax = axes[1, 2]
    panel_title(ax, "f", "Noise $\\sigma=0.4$")
    noise_groups = sorted([g for g in result["noise_participation_sweep"] if g["name"].startswith("noise_0.4_")], key=lambda g: g["M"])
    ms = [g["M"] for g in noise_groups]
    for method in methods:
        values = np.asarray([[g["runs"][j]["methods"][method]["random_prefix_false_split_fraction"][0]
                              for g in noise_groups] for j in range(20)])
        mean, lo, hi = mean_ci(values)
        # These are continuous time fractions, not independent Bernoulli updates.
        method_line(ax, ms, mean, lo, hi, method, band=False)
    ax.set_xlabel("Clients $M$")
    ax.set_ylabel("Split time fraction")
    ax.set_xticks(ms)
    ax.set_ylim(-.04, .56)
    ax.set_yticks([0, .25, .5])
    ax.margins(x=.08)
    legend = figure_legend(fig, axes, methods)
    caption = (
        "Current-gradient mechanism experiments with 20 paired runs per configuration. "
        "Panels (a-c) show the mean pre-update time average of the squared full deterministic "
        "proximal-gradient mapping versus training client-block gradient calls; bands are "
        "nominal pointwise 95% Student-t confidence intervals for the run mean (19 degrees of freedom), "
        "not simultaneous bands, run SD, or theoretical gradient-estimator variance. "
        "Panel (a) starts with a large common gradient and small disagreement; panel (b) starts with a zero "
        "common gradient and larger disagreement, retaining the reverse comparison. Each method's "
        "mapping is evaluated at its own evolving iterates. "
        "Panel (c) uses the original nonconvex data with M=2. Panels (d,e) show run probabilities "
        "of any shared-block split and final heterogeneous-block fusion under the threshold sweep; "
        "bars are Wilson 95% intervals from 20 independent run indicators. The vertical dotted "
        "line marks the sufficient shared-block threshold tau=0.14 and has no fusion interpretation. "
        "Panel (f) shows the pre-update shared-block split time fraction at observation SD 0.4, "
        "M*T=2400, and 32 samples per client; bars are pointwise Student-t intervals of the 20 "
        "run fractions, not intervals treating correlated updates as independent trials. "
        "Panels (a-e) vary algorithmic sampling on fixed data; panel (f) varies both training data "
        "and independent algorithmic sampling. Numerical split/fusion tolerance is 1e-10. "
        "Paired schedules and data are shared between methods; displayed intervals are for "
        "individual method means. Diagnostic full gradients and preprocessing are excluded "
        "from the training-call axis and are accounted for separately in the archived protocol."
    )
    return save_figure(fig, axes, legend, "mechanism_publication", out,
                       "Geometric sampling: stationarity and structural mechanisms", caption)


def baselines(result, out, width_mm, height_mm):
    methods = tuple(result["methods"])
    cases = {case["name"]: case for case in result["cases"]}
    ordered = [cases["common_potential_nonconvex"], cases["cross_block_nonconvex_stress"]]
    fixed_steps = {float(case["selected_step"]) for case in ordered}
    if len(fixed_steps) != 1:
        raise ValueError("The baseline figure caption requires a common selected-block stepsize.")
    fixed_step = fixed_steps.pop()
    fig, axes = plt.subplots(1, 3, figsize=(width_mm * MM, height_mm * MM))
    fig.subplots_adjust(left=.12, right=.965, bottom=.235, top=.70, wspace=.65)
    stationarity_panel(axes[0], ordered[0], "Common model", "a", methods, budget_weighted=True)
    stationarity_panel(axes[1], ordered[1], "Coupled stress", "b", methods, budget_weighted=True)
    ax = axes[2]
    panel_title(ax, "c", "Final objective")
    pos = np.asarray([0, 1], dtype=float)
    for idx, method in enumerate(methods):
        values = np.asarray([[case["runs"][j]["methods"][method]["final_objective"]
                              for case in ordered] for j in range(20)])
        mean, lo, hi = mean_ci(values)
        style = STYLES[method]
        ax.errorbar(pos + (idx - 1) * .16, mean, yerr=np.stack([mean - lo, hi - mean]),
                    fmt=style["marker"], color=style["color"], mfc="white", mew=.85,
                    capsize=2, elinewidth=.85, ms=4, linestyle="none")
    ax.axhline(0, color=".6", lw=.6, zorder=0)
    ax.set_xticks(pos, ["Common\npotential", "Coupled\nstress"])
    ax.set_ylabel("$\\Phi_S(U^{\\rm final})$")
    ax.set_xlim(-.4, 1.4)
    ax.margins(y=.16)
    legend = figure_legend(fig, axes, methods, y=.985)
    caption = (
        "Additional method comparison on 32-client, four-block nonconvex problems with block "
        "dimension eight, M=8, and 20 paired runs per case. All methods receive a total budget of "
        "8000 training client-block gradient calls, including the gradient-table method's "
        "128-call initialization; diagnostic full gradients are excluded and separately "
        "accounted for. Panels (a,b) report the budget-weighted running mean of the squared full "
        "deterministic proximal-gradient mapping: initialization calls retain the initial-state "
        "mapping, and each selected-block update is weighted by its training gradient calls. "
        "Thus the same call axis and initial-cost accounting apply to all methods. "
        "The methods are Geometric, Inverse inclusion, and the implemented client gradient-table "
        "baseline; the latter is a gradient-memory reference method. Panel (c) shows each "
        "method's final empirical objective, to be compared within each case. "
        "Lines/points are run means; bands/bars are nominal pointwise 95% Student-t confidence intervals "
        "with 19 degrees of freedom. They are not run SD or simultaneous bands; intervals "
        "describe individual method means rather than paired method differences. Data and "
        "algorithmic schedules remain paired across methods. Both cases are synthetic, use the "
        f"same fixed selected-block stepsize {fixed_step:g} across methods, and include no per-method tuning. The coupled case is a stress "
        "experiment, and generated zero-tilt labels are not claims about the coupled population optimum."
    )
    return save_figure(fig, axes, legend, "baseline_publication", out,
                       "Additional nonconvex comparisons at equal gradient-call budgets", caption)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results.json")
    parser.add_argument("--benchmark", type=Path, default=HERE / "benchmark_results.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--width-mm", type=float, default=130)
    parser.add_argument("--mechanism-height-mm", type=float, default=85)
    parser.add_argument("--baseline-height-mm", type=float, default=50)
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    font = configure_fonts()
    source_paths = [args.results]
    entries = {"mechanism_publication": mechanisms(json.loads(args.results.read_text(encoding="utf-8")),
               args.output, args.width_mm, args.mechanism_height_mm)}
    if not args.skip_benchmark:
        if not args.benchmark.exists():
            raise FileNotFoundError(f"New benchmark archive not yet available: {args.benchmark}; use --skip-benchmark for the original mechanism figure.")
        source_paths.append(args.benchmark)
        entries["baseline_publication"] = baselines(json.loads(args.benchmark.read_text(encoding="utf-8")),
               args.output, args.width_mm, args.baseline_height_mm)
    manifest = {
        "script": "geometry_sampling/plot_publication.py", "script_sha256": sha256(__file__),
        "reproduce": "python geometry_sampling/plot_publication.py",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "sources": [{"path": os.path.relpath(p, HERE.parent).replace("\\", "/"), "sha256": sha256(p)} for p in source_paths],
        "plot_parameters": {"font": font, "base_font_pt": 7.5, "tick_font_pt": 7.2,
            "line_width_pt": 1.2, "png_dpi": 600, "pdf_font_type": 42,
            "color_palette": {m: s["color"] for m, s in STYLES.items()},
            "global_legends": "reserved top region, outside all data axes",
            "logarithmic_panels": "both axes: mechanism (a-c), baseline (a-b)",
            "continuous_error": "pointwise mean +/- t(19,.975)*sample_SD/sqrt(20)",
            "bernoulli_error": "Wilson 95% interval, n=20 independent run indicators",
            "confidence_bands_are_simultaneous": False, "intervals_are_paired_differences": False,
            "source_numeric_archives_modified": False},
        "figures": entries,
    }
    (args.output / "publication_figures_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    captions = "\n\n".join(f"{name}\n{'=' * len(name)}\n{entry['caption']}" for name, entry in entries.items())
    (args.output / "publication_captions.txt").write_text(captions + "\n", encoding="utf-8")
    (args.output / "README.txt").write_text(
        "PUBLICATION FIGURES\n\nReproduce from the project root:\n"
        "  python geometry_sampling/plot_publication.py\n\n"
        "Dependencies: NumPy and Matplotlib. Numeric archives are read only.\n"
        "PDF: vector graphics with embedded TrueType fonts. SVG: editable vector output.\n"
        "PNG: 600 dpi at the recorded final physical size. Do not enlarge raster output.\n"
        "Both global legends occupy a dedicated region above all data panels.\n"
        "Continuous-data intervals are pointwise 95% Student-t intervals for 20 run means.\n"
        "Binary run-event intervals are Wilson 95% intervals. No simultaneous confidence\n"
        "coverage is claimed. The old archived summary SD remains unchanged.\n"
        "Captions and source checksums: publication_captions.txt / publication_figures_manifest.json.\n",
        encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "figures": {k: {"size_mm": v["size_mm"], "files": list(v["files"])} for k, v in entries.items()}}, indent=2))


if __name__ == "__main__":
    main()
