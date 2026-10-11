# Experimental protocols and resource accounting

This document records the settings, cost definitions, and interpretation of the experiments in *Geometry-Consistent Sampling for Nonconvex Personalized Optimization*. It includes the detailed allocation protocol moved from the manuscript appendix. Repository: [Geometry-Consistent Sampling for Nonconvex Personalized Optimization](https://github.com/zzh816/Geometry-Consistent-Sampling-for-Nonconvex-Personalized-Optimization).

All paths below are relative to the repository root. See the root README for installation and execution commands. Numerical results are retained for every declared seed, including unfavorable comparisons. No method-specific parameter tuning or seed filtering is applied.

## 1. Common objective, updates, and diagnostics

There are $N$ clients and $B$ parameter blocks. With the average-client metric $\|U\|_N^2=N^{-1}\sum_i\|u_i\|^2$, the regularizer is

$$
r(U)=\sum_{b=1}^B\tau_b\|DU_b\|_N,
\qquad (DU_b)_i=u_{i,b}-\frac1N\sum_j u_{j,b}.
$$

At each update, a block $b$ is drawn with probability $p_b$ and an independent uniform subset $S$ of $M\ge2$ clients is sampled. One training oracle call means one exact, current client-block empirical gradient. The selected step is $h_b=\gamma/p_b$. Geometric sampling uses

$$
\widetilde g_{i,b}=\bar g_{S,b}
+\frac{N-1}{M-1}\mathbf1_{\{i\in S\}}(g_{i,b}-\bar g_{S,b}),
\qquad \bar g_{S,b}=\frac1M\sum_{i\in S}g_{i,b}.
$$

The inverse-inclusion baseline (HT) uses $(N/M)\mathbf1_{\{i\in S\}}g_{i,b}$. Both apply the same full selected-block proximal operation to all clients. For $y=(y_i)$, its closed form is

$$
\operatorname{prox}_{h\tau_b\|D\cdot\|_N}(y)
=\bar y+\left(1-\frac{h\tau_b}{\|Dy\|_N}\right)_+Dy,
$$

with output $\bar y$ if $\|Dy\|_N=0$. All unselected blocks remain unchanged. Each method evaluates gradients at its own current state.

The stationarity diagnostic is the **squared deterministic full proximal-gradient mapping**

$$
\|\Gamma_\mu(U)\|_N^2
=\left\|\frac{U-\operatorname{prox}_{\mu r}(U-\mu\nabla F(U))}{\mu}\right\|_N^2,
\qquad \mu=\frac1{2L}.
$$

It is neither a sampled update norm nor the Moreau-envelope gradient. Its pre-update average $T^{-1}\sum_{t<T}\|\Gamma_\mu(U^t)\|_N^2$ corresponds to a uniformly sampled pre-update iterate under a constant base step. Final-iterate mappings are reported separately.

The cost-prefix average additionally charges initialization or pilot calls at the initial state:

$$
\overline G_{\mathrm{cost}}=
\frac{Q_0\|\Gamma_\mu(U^0)\|_N^2+
M\sum_{t<T}\|\Gamma_\mu(U^t)\|_N^2}{Q_0+MT},
$$

where $Q_0$ is the initial paid cost. This is a resource-weighted diagnostic, distinct from the uniform random-prefix measure. Both are recorded where applicable.

Full-gradient diagnostics, objectives, population/held-out metrics, and preprocessing are separate from training calls. Diagnostic gradients are never supplied to the estimator, gradient table, or probability scores. Numerical splitting and fusion use centered radius $>10^{-10}$ and $\le10^{-10}$, respectively. An “any split” event includes the final state; random-prefix split fractions average pre-update states, including initial consensus. These tolerance-based diagnostics do not identify an exact mathematical zero event.

## 2. Mechanism and regularization experiments

Implementation: `geometry_sampling/experiments.py`, orchestrated by `geometry_sampling/verify.py`. Results: `geometry_sampling/results.json`; original algebraic and deferred-update checks are retained in `geometry_sampling/fixtures/mechanism_reference.json`.

Each randomized configuration uses 20 paired runs. Within a pair, methods share data, initialization, block draws, client subsets, objective, steps, and proximal map. Independent schedules are generated before the runs; their hashes are stored. Full participation is checked to produce identical trajectories.

### 2.1 Exact subset enumeration and quadratic trajectories

For $N=8$, enumerate all size-$M$ subsets for $M\in\{2,4,8\}$ and gradients $2$, $2+0.1s_i$, and $0.8s_i$, where $s_i$ alternates between $-1$ and $1$. At $M=2$, the geometric/HT variances are respectively $0/12$, $0.06429/12.03$, and $4.114/1.92$. The last configuration intentionally lies outside the variance-advantage region.

The corresponding quadratic optimization runs use $M=2$, $T=1600$, selected step $h=0.03$, $\tau=0.12$, and zero initialization. There is no data randomness. Algorithm seeds are 20000–20019. The initial gradients agree with the enumeration vectors up to sign. A same-state variance ordering does not imply an ordering along different optimization trajectories.

### 2.2 Scalar-block nonconvex model

Use the common potential $h(x)=x^2/2+10(1-\cos x)$, $N=8$, $B=4$, 32 observations per client, and Gaussian observation standard deviation 0.12. The common mean is $(0.3,-0.2,0.1,0.25)$ and the centered client tilts are $(0,0,\pm0.8,\pm0.5)$. Set $\tau_b=0.25$, initial entries 0.4, $p_b=1/4$, $L=11$, and $\gamma=0.0025252525252525255$.

Participation sizes $M=2,4,8$ use $T=3200,1600,800$, respectively, for 6400 training calls in every case. The data are fixed and preserved from legacy seed 20261006, after that script's earlier algebraic/deferred-update checks; the full observations are archived. Update seeds are 21000–21019. Earlier five-run legacy outputs use a different horizon-dependent step schedule and should not be mixed with these constant-step results.

At $M=2,4$, geometric sampling has an observed shared-block split in 0/20 runs, versus 20/20 for HT. Neither method finally fuses a heterogeneous block. Full participation yields identical trajectories.

### 2.3 Threshold and observation-noise sweeps

Both use two scalar quadratic blocks, population mean $(1.5,0)$, alternating heterogeneous tilts $\pm0.5$ in the second block, selected step $h=0.015$, $p_b=1/2$, and zero initialization.

| Configuration | Settings | Randomness |
| --- | --- | --- |
| Threshold sweep | $N=8$, $M=2$, $T=1200$; $\tau\in\{0,0.05,0.14,0.25,0.6,1\}$; fixed errors $\pm0.04$; sharing bound 0.14 | Update seeds 22000–22019; fixed data |
| Noise/participation sweep | 32 observations/client; observation SD in $\{0,0.12,0.4\}$; $M\in\{2,4,8\}$; $\tau=0.25$; $MT=2400$ | Data seed $31000+j$ and independent update seed $23000+j$, $j=0,\ldots,19$ |

In the noise sweep, the same standard Gaussian observations are scaled by the chosen SD and then held fixed throughout training. Each pair shares its data and update draws. Excessive regularization can fuse a heterogeneous block; the resulting negative examples are retained.

### 2.4 Archived coupled quadratic certificate check

The quadratic Hessian is $Q=\begin{pmatrix}1&0.2\\0.2&1\end{pmatrix}$, the common mean is $(1.5,0)$, second-coordinate tilts are $\pm0.5$, and errors are $(\pm0.02,0)$. Use $N=8$, $M=2$, $T=1600$, $h=0.015$, $p_b=1/2$, zero initialization, and update seeds 24000–24019.

Set $\tau_y=0.25$, $c_M=7$, $C_M=3.5$, $A_y=0.5$, $E_x=0.02$, $R=c_MA_y^2/(2\tau_y)=3.5$, $\rho_y=\sqrt8R$, and $\tau_x=1.01C_M(0.2\rho_y+E_x)$. The selected step satisfies $hc_M=0.105\le1$. Recorded radii check the separately derived all-path certificate; they do not replace its proof. The experiment concerns departure from certified first-block consensus. Zero direct first-block tilt does not imply a shared unregularized population optimum because $Q^{-1}$ mixes the coordinates. The archived `score_violations_at_shared` field compares a fixed public geometric upper bound with $\tau_x$; it is not an actual sampled-score diagnostic and is not an HT certificate.

### 2.5 Mechanism figures and accounting

`geometry_sampling/plot_publication.py` generates `geometry_sampling/figures/mechanism_publication.pdf` and `.png`. Panels (a–c) show pre-update mapping averages; (d) any shared-block split; (e) final heterogeneous-block fusion; (f) pre-update split fractions at observation SD 0.4. Curves/points are means across 20 paired runs. Continuous bands/bars are pointwise 95% $t_{19}$ intervals for the mean; binary run-event bars are Wilson 95% intervals. They are not simultaneous confidence bands or paired-difference intervals. Panels (a–e) use fixed data; panel (f) varies independent data and update seeds.

Training costs $MT$ current client-block gradients per method/run. The mechanism script evaluates full mappings at $T+1$ states and recomputes the final state, recording $NB(T+2)$ additional client-block derivative diagnostics. Conditional variance diagnostics reuse these full gradients. Original nonconvex preprocessing reads $8\times32\times4=1024$ scalar observations once; a two-block noise replicate reads $8\times32\times2=512$. Empirical means are sufficient statistics, so these synthetic runs do not repeatedly reread raw observations during training.

## 3. Multidimensional comparison with gradient history

Implementation: `geometry_sampling/benchmark.py`. Outputs: `geometry_sampling/benchmark_results.json`, `benchmark_results.csv`, and `benchmark_summary_export.csv`. The two publication figures are `geometry_sampling/figures/baseline_publication.pdf` and `.png`.

Use $N=32$, $B=4$, block dimension 8, $M=8$, and 20 independent paired data/update seeds. The common potential is

$$
h(u)=\frac12\|u\|^2+5\sum_j(1-\cos u_j)
+\frac12u^\top(C\otimes I_8)u.
$$

Each local empirical loss is $h(u)-\langle0.4\mathbf1+\theta_i+\bar\epsilon_i,u\rangle$. The centered tilt radii are $(0,0,0.85,1.2)$; nonzero tilt directions are drawn from centered Gaussian arrays and rescaled to these radii. Each client has 32 Gaussian observations with SD 0.04. Set $\tau=(0.15,0.15,0.25,0.30)$, all initial entries to 3.6, $p_b=1/4$, and selected step $h=0.015$.

| Case | Coupling and smoothness | Data seeds | Update seeds |
| --- | --- | --- | --- |
| Common-potential nonconvex | $C=0$, $L=6$ | 51000–51019 | 61000–61019 |
| Cross-block nonconvex stress | Symmetric cycle entries $C_{12}=C_{23}=C_{34}=C_{41}=0.4$, $L=6.8$ | 52000–52019 | 62000–62019 |

The third method is a client-component gradient table with estimator

$$
H_{i,b}+\frac NM\mathbf1_{\{i\in S\}}(g_{i,b}(U)-H_{i,b}).
$$

The old table is used for estimation, then the sampled entries are refreshed. This is a conditionally unbiased control-variate baseline; it is not an implementation of SCAFFOLD or the original shared-variable SAGA algorithm. It stores $32\times4\times8=1024$ extra double-precision entries (8192 bytes), excluding common state arrays.

All methods spend exactly 8000 training calls. Geometric and HT take 1000 updates. The table pays 128 initialization calls and performs 984 updates, using the corresponding prefix of the common 1000-draw schedule. Its cost-prefix mapping charges initialization at the initial mapping. Steps and thresholds have no method-specific tuning; $h\le1/(9L)$. Each full diagnostic uses all $NB$ block derivatives, so the recorded diagnostic count is $NB(T_{\mathrm{method}}+1)$. Preprocessing generates/reads 32768 scalar observations per data run, and all methods receive the same empirical sufficient statistics.

In the separable case, the data-only sharing certificate holds for all 20 datasets, with $C_M\bar r_\infty\le0.081<0.15$. Geometric sampling never splits a certified shared block, while HT and the table do so in all runs. Nevertheless, their respective mean final squared mappings are 0.0195, 0.0193, and $5.64\times10^{-8}$; geometric sampling also has worse final objectives. Its mean population-risk gain relative to broadcasting the same output mean is $-0.569$. A quadratic-subclass risk guarantee cannot be extended to this nonconvex instance.

In the coupled case, mean final mappings are 0.0293, 0.0248, and $3.18\times10^{-12}$. All methods depart from zero-tilt consensus; these signal labels do not describe the coupled population minimizer. The separately archived conservative quadratic certificate preserves consensus for both methods. These comparisons demonstrate conditional structural preservation and its limits, without establishing uniform algorithmic superiority.

The baseline figure reports cost-weighted mapping averages for both cases and final empirical objectives compared **within** each case. Means and pointwise 95% $t_{19}$ intervals use the 20 paired data/update runs.

## 4. Block-probability ablation with paid pilots

Scripts and outputs are in `geometry_sampling/allocation_revision/`. The methods are uniform probabilities, fixed pilot-based probabilities, and predictable exponential-moving-average probabilities (EMA). The same three strategies are used on the synthetic and WDBC cases below.

### 4.1 Observable scores and probability updates

For fixed-pilot and EMA, one size-$M$ subset is queried at the initial state for every block. These $BM$ pilot calls are charged before the first update. Their initial score is the unbiased finite-population estimate

$$
\widehat\zeta_b^2=
\frac{N-1}{N(M-1)}\sum_{i\in S}\|g_{i,b}-\bar g_{S,b}\|^2.
$$

With score $s_b$, both methods use

$$
p_b=\delta+(1-B\delta)
\frac{\sqrt{s_b+10^{-12}}}{\sum_j\sqrt{s_j+10^{-12}}}.
$$

The implementation clips scores below zero before taking square roots to guard against numerical roundoff. Fixed-pilot probabilities retain the initial scores. Following an update, EMA changes only the selected block's score to $0.9s_b+0.1\widehat\zeta_b^2$, reusing gradients already queried for that update. Other scores remain unchanged. The next probability vector therefore uses observed history only. Scores are estimates, not certified trajectory-wide variance bounds, and the rule is not asserted to solve the constrained oracle allocation problem.

EMA requires no extra gradient calls after initialization; its score and probability arithmetic are included in the training timer. Fixed-pilot and EMA share pilot subsets. All three methods share subsequent uniform block quantiles and node-subset sequences. Different probability vectors can select different blocks from the same quantile. Each uses the same $\gamma$ and $\delta$ within a case, with $h_{t,b}=\gamma/p_b^t$.

### 4.2 Separable nonconvex case

Reuse the separable model in Section 3: $N=32$, $B=4$, block dimension 8, $M=8$, $L=6$, $\mu=1/12$, the same initial state, thresholds, and data seeds 51000–51019. Independent ablation update seeds are 71000–71019. The base step is $\gamma=\delta/(12L)$ with $\delta=0.05$, so $h_{t,b}\le1/(12L)$. Each method has 32000 paid client-block calls. Uniform takes 4000 updates; fixed-pilot and EMA each pay 32 pilot calls and take 3996 updates. Parameters were fixed before the runs.

The declared squared-mapping targets are $1$, $10^{-1}$, and $10^{-2}$. Results and all first-hit/non-hit records are in `allocation_results.json`; `allocation_results.csv` and `allocation_summary.txt` provide tabular summaries.

### 4.3 Coupled logistic objective on WDBC

The public UCI Wisconsin Diagnostic Breast Cancer dataset contains 569 observations and 30 features. Its original data, attribution, license, and hash are documented in [DATA.md](DATA.md). Clients are **artificial task partitions**, not observed users, agents, or hospitals.

For each of 20 paired splits (seeds 81000–81019), shuffle each class separately and allocate the first $\lfloor0.75n_{\mathrm{class}}\rfloor$ observations to training. This gives 426 training and 143 held-out observations. Standardize each feature with its training-set mean and population standard deviation; then prepend an intercept. No test statistics are used for preprocessing.

Allocate the train and test pools separately to eight artificial clients by class. The first four clients collectively receive 80% of the malignant pool and 20% of the benign pool **in expectation**, equally weighted within that half; the last four receive the reversed proportions. Random class-specific allocations produce unequal client sizes. All train/test row assignments, client sizes, training malignant fractions, and fitted standardization parameters are recorded per seed.

For client $i$, use empirical binary logistic loss plus $0.05\|u_i\|^2/2$; the ridge penalty includes the intercept. With the intercept first, the 31 coordinates are split into dimensions $(11,10,10)$ (zero-based columns 0–10, 11–20, and 21–30). Set $\tau_b=0.04$, zero initialization, $M=4$, $\delta=0.1$, $\gamma=\delta/(12L)$, and

$$
L=\max_i\frac{\|X_i\|_2^2}{4n_i}+0.05,
$$

where $X_i$ contains the intercept column. Update seeds are 91000–91019. Each method receives 12000 paid calls. Uniform takes 3000 updates; nonuniform methods pay 12 pilot calls and take 2997 updates. Declared squared-mapping targets are $10^{-1}$, $10^{-2}$, and $10^{-3}$.

The objective is smooth and convex before the convex consensus penalty, hence belongs to the general stationarity setting. Local logistic losses couple the feature blocks and do not satisfy the common-potential structural model. No true sharing labels or structure-recovery claims are assigned to this case.

Held-out logistic losses exclude training regularization. Classification uses the sign of the logit (threshold zero). Both equal-client and sample-weighted pooled summaries are archived; the manuscript reports equal-client means. Results, row assignments, and preprocessing are in `wdbc_results.json`; `wdbc_results.csv` and `wdbc_summary.txt` contain summaries.

### 4.4 Reported outcomes and limitations

Mapping entries below are mean $\pm$ sample SD over 20 paired runs. “Reached” counts runs ever attaining squared mapping $\le10^{-2}$. Median calls are conditional on attaining it and must be read with the success count.

| Case | Probabilities | Final squared mapping | Cost-prefix mapping | Reached | Median paid calls |
| --- | --- | --- | --- | --- | --- |
| Nonconvex | Uniform | $0.0117\pm0.0336$ | $167.61\pm0.70$ | 20/20 | 21196 |
| Nonconvex | Fixed pilot | $0.0115\pm0.0363$ | $168.30\pm6.11$ | 18/20 | 20644 |
| Nonconvex | EMA | $0.00476\pm0.00751$ | $168.70\pm6.52$ | 19/20 | 20904 |
| WDBC | Uniform | $0.00687\pm0.00094$ | $0.1300\pm0.0070$ | 20/20 | 10048 |
| WDBC | Fixed pilot | $0.00687\pm0.00096$ | $0.1322\pm0.0072$ | 20/20 | 10002 |
| WDBC | EMA | $0.00688\pm0.00095$ | $0.1320\pm0.0070$ | 20/20 | 10024 |

For the nonconvex case, EMA has a smaller mean final mapping, but its paired EMA-minus-uniform descriptive 95% $t_{19}$ interval is $[-0.0233,0.00950]$. This does not establish a uniform advantage. All strategies preserve the two certified shared blocks in every run and have zero final false fusion on the two nonzero-tilt blocks. First crossings need not persist: the numbers still below $10^{-2}$ at the final iterate are 19, 18, and 18. At target $10^{-1}$, all 20 runs cross for each strategy, with median costs 18528, 18476, and 18496 calls. The results archive also retains the declared target 1.

On WDBC, no method reaches $10^{-3}$ within budget. Equal-client held-out losses are $0.15226\pm0.02501$, $0.15221\pm0.02504$, and $0.15226\pm0.02503$ for uniform, fixed-pilot, and EMA; classification errors are 3.21%, 3.21%, and 3.26%. The EMA-minus-uniform interval for held-out loss is $[-2.23,2.28]\times10^{-4}$. Its cost-prefix mapping is worse by 0.00196, with interval $[0.00104,0.00288]$.

These runs demonstrate implementability under common resource accounting, not a convergence-rate or generalization improvement. The intervals are descriptive paired-mean intervals without multiple-comparison adjustment. First hits use external full-gradient diagnostics; they are neither free stopping certificates nor assurances that a target remains satisfied.

### 4.5 Resource accounting and timing

Every queried logistic block gradient still needs a forward product with **all** feature coordinates. The implementation records training-example reads as well as client-block oracle calls because client sample sizes differ. Each WDBC strategy reads about $6.39\times10^5$ local training examples across all paid queries, including pilots. The exact per-run counts are saved. These counts do not turn a block derivative into an isolated partial forward pass.

The following are recorded mean times in seconds for the dense single-process prototype, in method order uniform/fixed-pilot/EMA:

| Case | Uniform training | Fixed-pilot training | EMA training | Separately timed diagnostics |
| --- | --- | --- | --- | --- |
| Nonconvex | 0.121 | 0.122 | 0.163 | approximately 0.193–0.194 |
| WDBC | 0.211 | 0.212 | 0.246 | approximately 0.604–0.609 |

Training timers include pilot work and probability updates; diagnostics are timed separately and excluded from the training-call axis. Model/data construction and preprocessing are outside those update timers. These small local timings are not distributed communication measurements or large-model throughput benchmarks and will vary with the execution environment.

`make_allocation_figure.py` reads both result files and writes `allocation_comparison.png`. It plots median full squared mappings against paid calls, with interquartile shaded regions across 20 runs. Those regions are not confidence bands. Pilots are on the horizontal cost axis; additional full-gradient diagnostic work is separately recorded.

## 5. Reproducibility records

The reported allocation runs used Python 3.12.14 and NumPy 2.3.5; the allocation figure used Matplotlib 3.11.2. Scripts record runtime versions, source checksums, all paired outcomes, and cost counters. A full second synthetic allocation run matched numerical SHA256 after timing fields were removed. The first WDBC paired seed was independently replayed for all three strategies, with matching numerical fields after removing timings. Repeated runs should compare numerical fields separately from nondeterministic clock measurements.

The mechanism fixture additionally contains the archived exact subset-enumeration checks, 500-round deferred-affine/moment checks, and 300-state retention/paired-risk checks. The supplied `reproducibility.json` records the original numerical and manuscript-rendering audit; its historical source and LaTeX hashes document that audit and are not dependencies. The standalone package no longer generates manuscript LaTeX. Its `verify.py` regenerates numerical JSON and records replay checks against the packaged sources. Main publication figures use the archived numerical outcomes, and the allocation experiments extend rather than replace the geometric/HT/gradient-table comparison.

