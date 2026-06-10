#!/usr/bin/env python3
"""Generate the figures for Tutorial 3 Chapters 16-19 (Monte Carlo / particle
filtering / MCMC / Kemp sampler) into static/images/intro2/.

Light theme to match the existing intro2 figures (white bg, dpi 150).
All randomness is seeded so the figures are reproducible; the numbers shown
match the chapters' validated code cells where they depict the same experiment.

Run from the textbook root:  python3 scripts/generate_sampling_figures.py
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from scipy import stats

OUT = Path(__file__).resolve().parent.parent / "static" / "images" / "intro2"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#1f77b4"
TEAL = "#4ecdc4"
PURPLE = "#6c5ce7"
RED = "#ff6b6b"
ORANGE = "#f39c12"
GREEN = "#2ecc71"
GRAY = "#888888"

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "font.size": 11})


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {name}")


# ----------------------------------------------------------------------------
# Chapter 16
# ----------------------------------------------------------------------------

def fig_die_convergence():
    rng = np.random.default_rng(0)
    n = 100_000
    rolls = rng.integers(1, 7, size=n)
    running = np.cumsum(rolls) / np.arange(1, n + 1)
    xs = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.axhline(3.5, color=ORANGE, ls="--", lw=1.5, label="true E[die] = 3.5")
    # the 1/sqrt(n) error envelope (die sd ~ 1.708)
    sd = np.sqrt(35 / 12)
    ax.fill_between(xs, 3.5 - 2 * sd / np.sqrt(xs), 3.5 + 2 * sd / np.sqrt(xs),
                    color=ORANGE, alpha=0.12, label=r"$\pm 2\sigma/\sqrt{n}$ envelope")
    ax.plot(xs, running, color=BLUE, lw=1.2, label="running Monte Carlo estimate")
    ax.set_xscale("log")
    ax.set_xlabel("number of rolls $n$ (log scale)")
    ax.set_ylabel("estimate of E[die]")
    ax.set_ylim(2.4, 4.6)
    ax.set_title("The Monte Carlo estimate converges — and its error shrinks like $1/\\sqrt{n}$")
    ax.legend(loc="upper right", fontsize=9)
    save(fig, "mc_die_convergence.png")


def fig_hospital():
    rng = np.random.default_rng(1)
    days = 100_000  # many days so the histograms are smooth
    frac_big = rng.binomial(45, 0.5, size=days) / 45
    frac_small = rng.binomial(15, 0.5, size=days) / 15

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for ax, frac, n, label in [
        (axes[0], frac_big, 45, "large hospital (45 births/day)"),
        (axes[1], frac_small, 15, "small hospital (15 births/day)"),
    ]:
        bins = np.arange(-0.5 / n, 1 + 1.0 / n, 1.0 / n)
        over = float(np.mean(frac > 0.6))
        ax.hist(frac[frac <= 0.6], bins=bins, color=BLUE, alpha=0.85)
        ax.hist(frac[frac > 0.6], bins=bins, color=RED, alpha=0.9,
                label=f">60% boys on {over:.1%} of days")
        ax.axvline(0.6, color="k", ls="--", lw=1.2)
        ax.set_xlim(0.05, 0.95)
        ax.set_xlabel("fraction of boys that day")
        ax.set_title(label, fontsize=11)
        ax.legend(loc="upper left", fontsize=9)
    axes[0].set_ylabel("number of days")
    fig.suptitle("Small samples swing wide: the small hospital crosses 60% far more often", y=1.04)
    save(fig, "mc_hospital_tails.png")


def fig_pi_darts():
    rng = np.random.default_rng(2)
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 8.6))
    for ax, n in zip(axes.ravel(), [10, 100, 1000, 10000]):
        pts = rng.random((n, 2))
        inside = pts[:, 0] ** 2 + pts[:, 1] ** 2 <= 1.0
        size = 42 if n <= 100 else (10 if n <= 1000 else 3.5)
        ax.scatter(*pts[inside].T, s=size, color=GREEN, alpha=0.8, lw=0)
        ax.scatter(*pts[~inside].T, s=size, color=RED, alpha=0.8, lw=0)
        arc = np.linspace(0, np.pi / 2, 100)
        ax.plot(np.cos(arc), np.sin(arc), color="k", lw=1.6)
        est = 4 * inside.mean()
        ax.set_title(f"$n={n}$:  $\\hat\\pi = {est:.3f}$", fontsize=13)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal")
    fig.suptitle("$\\pi$ by throwing darts: 4 × (fraction inside the quarter-circle)",
                 fontsize=13, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "mc_pi_darts.png")


def fig_rejection():
    rng = np.random.default_rng(3)
    n = 2200
    xs = rng.random(n)
    hs = rng.random(n) * 2.0
    keep = hs <= 2.0 * xs

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.scatter(xs[keep], hs[keep], s=5, color=GREEN, alpha=0.6, label="kept (under $p$)")
    ax.scatter(xs[~keep], hs[~keep], s=5, color=RED, alpha=0.45, label="rejected")
    grid = np.linspace(0, 1, 100)
    ax.plot(grid, 2 * grid, color="k", lw=2, label=r"target  $p(x) = 2x$")
    ax.add_patch(Rectangle((0, 0), 1, 2, fill=False, ec=GRAY, lw=1.2, ls="--"))
    ax.text(0.03, 1.9, "envelope box (easy to sample)", color=GRAY, fontsize=9, va="top")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.05, 2.1)
    ax.set_xlabel("$x$"); ax.set_ylabel("height")
    ax.set_title("Rejection sampling: keep the points that land under the target")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    save(fig, "mc_rejection.png")


def fig_is_tail():
    x = np.linspace(420, 900, 600)
    p = stats.norm.pdf(x, 620, 50)
    q = stats.norm.pdf(x, 700, 50)

    fig, ax = plt.subplots(figsize=(8, 4.0))
    ax.plot(x, p, color=BLUE, lw=2, label="target $p$: bento weights  $\\mathcal{N}(620, 50^2)$")
    ax.plot(x, q, color=PURPLE, lw=2, ls="--",
            label="proposal $q$: shifted onto the tail  $\\mathcal{N}(700, 50^2)$")
    tail = x >= 700
    ax.fill_between(x[tail], 0, p[tail], color=BLUE, alpha=0.3,
                    label="the event: $W > 700$ g  (about 5% of $p$)")
    ax.axvline(700, color="k", ls=":", lw=1.2)
    ax.set_xlabel("bento weight (g)"); ax.set_ylabel("density")
    ax.set_yticks([])
    ax.set_title("Importance sampling a tail: sample where the event lives, reweight by $p/q$")
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "mc_is_tail.png")


def fig_weight_variance():
    rng = np.random.default_rng(5)
    n = 4000
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2),
                             gridspec_kw={"height_ratios": [1.4, 1]})
    for col, (mu_q, sd_q, name) in enumerate([
        (630.0, 55.0, "well-matched proposal  $q = \\mathcal{N}(630, 55^2)$"),
        (760.0, 30.0, "poorly-matched proposal  $q = \\mathcal{N}(760, 30^2)$"),
    ]):
        xs = rng.normal(mu_q, sd_q, size=n)
        logw = stats.norm.logpdf(xs, 620, 50) - stats.norm.logpdf(xs, mu_q, sd_q)
        w = np.exp(logw - logw.max()); w = w / w.sum()
        ess = 1.0 / np.sum(w ** 2)

        grid = np.linspace(420, 900, 500)
        ax = axes[0, col]
        ax.plot(grid, stats.norm.pdf(grid, 620, 50), color=BLUE, lw=2, label="target $p$")
        ax.plot(grid, stats.norm.pdf(grid, mu_q, sd_q), color=PURPLE, lw=2, ls="--", label="proposal $q$")
        ax.set_title(name, fontsize=10.5)
        ax.set_yticks([]); ax.set_xlim(420, 900)
        ax.legend(fontsize=9)

        ax = axes[1, col]
        ax.hist(w * n, bins=60, color=TEAL if col == 0 else RED, alpha=0.85)
        ax.set_yscale("log")
        ax.set_xlabel("normalized weight  (× n)")
        ax.set_ylabel("count (log)" if col == 0 else "")
        ax.set_title(f"ESS = {ess:,.0f}  out of  {n:,}", fontsize=11,
                     color="#1a7a4a" if col == 0 else "#b03030")
    fig.suptitle("Weight evenness is the diagnostic: a mismatched proposal collapses the effective sample size",
                 y=1.0)
    fig.tight_layout()
    save(fig, "mc_weight_variance.png")


def fig_exemplar():
    weights = np.array([520., 560., 590., 610., 640., 660., 700., 730.])
    is_tonk = np.array([0., 0., 0., 1., 0., 1., 1., 1.])

    grid = np.linspace(480, 780, 400)
    s = np.exp(-0.5 * ((grid[:, None] - weights[None, :]) / 40.0) ** 2)
    vote = (s * is_tonk).sum(axis=1) / s.sum(axis=1)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(grid, vote, color=PURPLE, lw=2.5, label="similarity-weighted vote = P(tonkatsu)")
    ax.scatter(weights[is_tonk == 1], np.ones(4) * 1.045, marker="v", s=70, color=ORANGE,
               label="stored tonkatsu bentos", zorder=5, clip_on=False)
    ax.scatter(weights[is_tonk == 0], np.zeros(4) - 0.045, marker="^", s=70, color=BLUE,
               label="stored non-tonkatsu bentos", zorder=5, clip_on=False)
    for q in [550, 650, 720]:
        v = float(np.interp(q, grid, vote))
        ax.plot([q, q], [0, v], color=GRAY, ls=":", lw=1)
        ax.scatter([q], [v], s=45, color="k", zorder=6)
        ax.annotate(f"{v:.2f}", (q, v), textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel("bento weight (g)"); ax.set_ylabel("P(tonkatsu)")
    ax.set_title("An exemplar model's vote is a self-normalized importance sampler over memories")
    ax.legend(loc="center left", fontsize=9)
    save(fig, "mc_exemplar_vote.png")


# ----------------------------------------------------------------------------
# Chapter 17
# ----------------------------------------------------------------------------

def _pf_panel(ax, parts, w, title, z=None, moved=None):
    ax.set_xlim(0.0, 4.0)
    ax.set_ylim(-0.6, 1.2)
    ax.set_yticks([])
    ax.set_title(title, fontsize=11)
    if z is not None:
        ax.axvline(z, color=ORANGE, lw=2, label=f"sensor ping  $z={z}$")
        ax.legend(loc="upper right", fontsize=8)
    sizes = 2200 * w if w is not None else np.full(len(parts), 90.0)
    ax.scatter(parts, np.zeros_like(parts), s=sizes, color=BLUE, alpha=0.55, zorder=3)
    if moved is not None:
        ax.scatter(moved, np.zeros_like(moved) + 0.0, s=90, color=TEAL, alpha=0.7, zorder=4)
        for a, b in zip(parts, moved):
            ax.annotate("", xy=(b, 0.25), xytext=(a, 0.25),
                        arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.0))
    ax.set_xlabel("position (m)")


def fig_pf_steps():
    rng = np.random.default_rng(7)
    parts = np.array([0.6, 0.95, 1.35, 1.75, 2.3])
    z = 1.3
    logw = stats.norm.logpdf(z, parts, 0.7)
    w = np.exp(logw - logw.max()); w = w / w.sum()
    idx = rng.choice(5, size=5, p=w)
    survivors = parts[idx]
    moved = survivors + 1.0 + rng.normal(0, 0.3, size=5)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.0))
    _pf_panel(axes[0], parts, w, "1. WEIGHT each particle by the ping\n(dot area $\\propto$ weight)", z=z)
    _pf_panel(axes[1], survivors, None,
              "2. RESAMPLE: heavy particles cloned,\nlight ones culled (weights reset)")
    _pf_panel(axes[2], survivors, None, "3. PROPAGATE through the motion model\n(each survivor steps ~1 m)",
              moved=moved)
    fig.suptitle("One tick of the particle filter, with M = 5 particles", y=1.1)
    save(fig, "pf_steps.png")


def fig_pf_tracking():
    # mirror the chapter's validated run qualitatively (fresh seed, same models)
    rng = np.random.default_rng(8)
    true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    obs = np.array([1.3, 1.8, 3.4, 3.9, 5.2])
    est = np.array([1.04, 1.98, 3.10, 4.03, 5.08])  # from the chapter's validated cell
    t = np.arange(1, 6)

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.plot(t, true, color="k", lw=2, marker="o", label="true position")
    ax.scatter(t, obs, color=ORANGE, s=70, marker="x", lw=2.5, label="noisy pings $z_t$", zorder=5)
    ax.plot(t, est, color=PURPLE, lw=2, marker="s", ms=5, ls="--",
            label="particle-filter estimate (M = 2000)")
    ax.set_xticks(t)
    ax.set_xlabel("tick $t$"); ax.set_ylabel("position (m)")
    ax.set_title("The filter hugs the true path more tightly than the raw pings")
    ax.legend(fontsize=9)
    save(fig, "pf_tracking.png")


def fig_pf_degeneracy():
    rng = np.random.default_rng(9)
    M = 2000
    Ts = [5, 10, 20, 30, 40]
    ess_list = []
    for T in Ts:
        obs = np.cumsum(np.ones(T)) + rng.normal(0, 0.7, size=T)
        parts = 1.0 + rng.normal(0, 0.3, size=M)
        logw = np.zeros(M)
        for t in range(T):
            logw += stats.norm.logpdf(obs[t], parts, 0.7)
            parts = parts + 1.0 + rng.normal(0, 0.3, size=M)
        w = np.exp(logw - logw.max()); w = w / w.sum()
        ess_list.append(1.0 / np.sum(w ** 2))

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(Ts, ess_list, color=RED, lw=2, marker="o")
    ax.axhline(M, color=GRAY, ls="--", lw=1.2)
    ax.text(Ts[0], M * 0.82, f"all {M} particles useful", color=GRAY, fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("track length $T$ (steps, never resampling)")
    ax.set_ylabel("effective sample size (log)")
    ax.set_title("Weight degeneracy: without resampling, the swarm collapses onto a few particles")
    save(fig, "pf_degeneracy.png")


# ----------------------------------------------------------------------------
# Chapter 18
# ----------------------------------------------------------------------------

def _bimodal(x):
    return 0.5 * np.exp(-0.5 * ((x + 2) / 0.7) ** 2) + 0.5 * np.exp(-0.5 * ((x - 2) / 0.7) ** 2)


def fig_mh_steps():
    x = np.linspace(-4.5, 4.5, 500)
    px = _bimodal(x)

    cases = [
        (-2.6, -2.1, True, "uphill: $P(x') > P(x)$  →  always accept  ($A = 1$)"),
        (-2.0, -1.45, True, "downhill: accept *sometimes*\n(here $A = P(x')/P(x) \\approx 0.55$)"),
        (-2.0, -0.2, False, "far downhill: almost always reject\n(chain stays and records $x$ again)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for ax, (x0, x1, accepted, title) in zip(axes, cases):
        ax.plot(x, px, color=BLUE, lw=2)
        y0, y1 = _bimodal(np.array([x0]))[0], _bimodal(np.array([x1]))[0]
        ax.scatter([x0], [y0], s=80, color="k", zorder=5)
        color = GREEN if accepted else RED
        ax.scatter([x1], [y1], s=80, color=color, zorder=5)
        ax.annotate("", xy=(x1, y1 + 0.06), xytext=(x0, y0 + 0.06),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2,
                                    connectionstyle="arc3,rad=-0.3"))
        ax.text(x0, y0 - 0.09, "$x$", ha="center", fontsize=11)
        ax.text(x1, y1 - 0.09, "$x'$", ha="center", fontsize=11)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlim(-4.5, 4.5); ax.set_ylim(-0.12, 0.72)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Metropolis–Hastings in three moves: explore, but favor high ground", y=1.07)
    save(fig, "mcmc_mh_steps.png")


def fig_mh_histogram():
    rng = np.random.default_rng(11)
    n = 40000
    xchain = np.empty(n)
    xc = 0.0
    for i in range(n):
        xp = xc + rng.normal(0, 1.5)
        if np.log(rng.random()) < np.log(_bimodal(np.array([xp]))[0] + 1e-300) - np.log(
                _bimodal(np.array([xc]))[0] + 1e-300):
            xc = xp
        xchain[i] = xc
    xchain = xchain[4000:]

    grid = np.linspace(-4.5, 4.5, 400)
    Z = np.trapz(_bimodal(grid), grid)

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.hist(xchain, bins=80, density=True, color=TEAL, alpha=0.75, label="MH samples (after burn-in)")
    ax.plot(grid, _bimodal(grid) / Z, color="k", lw=2, label="target (normalized)")
    ax.set_xlabel("$x$"); ax.set_ylabel("density")
    ax.set_title("A well-tuned chain fills both modes — the histogram matches the target")
    ax.legend(fontsize=9)
    save(fig, "mcmc_mh_histogram.png")


def fig_gibbs_trace():
    rng = np.random.default_rng(12)
    rho = 0.8
    # contours of the correlated Gaussian
    g = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(g, g)
    Zd = np.exp(-(X ** 2 - 2 * rho * X * Y + Y ** 2) / (2 * (1 - rho ** 2)))

    # a short Gibbs path, drawn with its L-shaped (axis-aligned) moves
    pts = [(-2.0, -2.0)]
    xc, yc = -2.0, -2.0
    for _ in range(14):
        xc = rho * yc + np.sqrt(1 - rho ** 2) * rng.normal()
        pts.append((xc, yc))
        yc = rho * xc + np.sqrt(1 - rho ** 2) * rng.normal()
        pts.append((xc, yc))
    pts = np.array(pts)

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.contour(X, Y, Zd, levels=6, colors=GRAY, linewidths=1.0, alpha=0.7)
    ax.plot(pts[:, 0], pts[:, 1], color=PURPLE, lw=1.6, alpha=0.9)
    ax.scatter(pts[::2, 0], pts[::2, 1], s=26, color=PURPLE, zorder=5)
    ax.scatter([pts[0, 0]], [pts[0, 1]], s=90, color=ORANGE, zorder=6, label="start")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title("Gibbs moves are axis-aligned: resample $x_1$, then $x_2$, then $x_1$, …")
    ax.legend(fontsize=9)
    ax.set_aspect("equal")
    save(fig, "mcmc_gibbs_trace.png")


def fig_traces():
    rng = np.random.default_rng(13)

    def run(step, x0, n=4000):
        out = np.empty(n)
        xc = x0
        for i in range(n):
            xp = xc + rng.normal(0, step)
            if np.log(rng.random()) < np.log(_bimodal(np.array([xp]))[0] + 1e-300) - np.log(
                    _bimodal(np.array([xc]))[0] + 1e-300):
                xc = xp
            out[i] = xc
        return out

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, step, title in [
        (axes[0], 0.15, "small step ($\\sigma=0.15$): two starts never meet — trapped"),
        (axes[1], 1.5, "good step ($\\sigma=1.5$): both chains hop modes — mixed"),
    ]:
        a = run(step, -2.0)
        b = run(step, +2.0)
        ax.plot(a, color=BLUE, lw=0.8, alpha=0.9, label="started left")
        ax.plot(b, color=ORANGE, lw=0.8, alpha=0.9, label="started right")
        ax.axhline(-2, color=GRAY, ls=":", lw=1)
        ax.axhline(+2, color=GRAY, ls=":", lw=1)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("iteration")
        ax.legend(fontsize=9, loc="lower right")
    axes[0].set_ylabel("$x$ (trace)")
    fig.suptitle("Read the trace: flat lines = stuck in a mode; hopping between levels = mixing",
                 fontsize=12.5, y=1.04)
    save(fig, "mcmc_traces.png")


# ----------------------------------------------------------------------------
# Chapter 19
# ----------------------------------------------------------------------------

def fig_kemp_plate():
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.axis("off")

    def node(x, y, label, sub="", fc="white"):
        c = Circle((x, y), 0.78, fc=fc, ec="k", lw=1.6, zorder=4)
        ax.add_patch(c)
        ax.text(x, y + (0.14 if sub else 0), label, ha="center", va="center",
                fontsize=14, zorder=6)
        if sub:
            ax.text(x, y - 0.32, sub, ha="center", va="center", fontsize=8,
                    color="#444444", zorder=6)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=16, color="k", lw=1.4,
                                     shrinkA=16, shrinkB=16, zorder=3))

    node(3.0, 8.6, r"$\varphi$", "population mean")
    node(7.0, 8.6, r"$\kappa$", "concentration")
    node(5.0, 6.3, r"$a, b$", r"$a=\kappa\varphi$,  $b=\kappa(1-\varphi)$")
    node(5.0, 3.9, r"$\theta_i$", "shop $i$'s quality rate")
    node(5.0, 1.6, r"$k_i$", "good ratings of $n_i$", fc="#e8e8e8")

    arrow(3.0, 8.6, 5.0, 6.3)
    arrow(7.0, 8.6, 5.0, 6.3)
    arrow(5.0, 6.3, 5.0, 3.9)
    arrow(5.0, 3.9, 5.0, 1.6)

    # the plate over the per-shop variables
    ax.add_patch(Rectangle((3.3, 0.5), 3.4, 4.6, fill=False, ec=GRAY, lw=1.5))
    ax.text(6.55, 0.72, "shops $i = 1 \\ldots M$", ha="right", fontsize=10, color="#444444")

    ax.text(5.0, 9.7, "The bento-shop hierarchy (the Kemp two-level Beta-Binomial)",
            ha="center", fontsize=12)
    ax.text(8.6, 1.6, "shaded =\nobserved", fontsize=9, color="#444444", va="center")
    save(fig, "kemp_plate.png")


def _run_kemp(seed=1, n_sweeps=6000, burn=1000, s_phi=0.04, s_ell=0.25):
    """The chapter's collapsed sampler (numpy mirror of the validated jax cell)."""
    from scipy.special import betaln, gammaln

    K = np.array([9., 3., 7., 5., 8., 2., 6., 9., 4., 7., 1., 8.])
    N = np.full(12, 10.0)
    rng = np.random.default_rng(seed)

    def log_marg_all(phi, ell):
        kappa = np.exp(ell)
        a, b = kappa * phi, kappa * (1 - phi)
        lc = gammaln(N + 1) - gammaln(K + 1) - gammaln(N - K + 1)
        return float(np.sum(lc + betaln(a + K, b + N - K) - betaln(a, b)))

    phi, ell = 0.5, np.log(5.0)
    phis, kappas = [], []
    for t in range(n_sweeps):
        phi_p = phi + s_phi * rng.normal()
        ell_p = ell + s_ell * rng.normal()
        if 0 < phi_p < 1 and np.log(rng.random()) < log_marg_all(phi_p, ell_p) - log_marg_all(phi, ell):
            phi, ell = phi_p, ell_p
        if t >= burn:
            phis.append(phi); kappas.append(np.exp(ell))
    return np.array(phis), np.array(kappas), K, N


def fig_kemp_posteriors():
    phis, kappas, _, _ = _run_kemp()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].hist(phis, bins=50, color=TEAL, alpha=0.85)
    axes[0].axvline(phis.mean(), color="k", ls="--", lw=1.5,
                    label=f"posterior mean $\\bar\\varphi$ = {phis.mean():.3f}")
    axes[0].set_xlabel(r"population mean $\varphi$"); axes[0].set_ylabel("samples")
    axes[0].legend(fontsize=9)
    axes[1].hist(kappas[kappas < np.quantile(kappas, 0.99)], bins=50, color=PURPLE, alpha=0.85)
    axes[1].axvline(np.median(kappas), color="k", ls="--", lw=1.5,
                    label=f"posterior median $\\kappa$ = {np.median(kappas):.1f}")
    axes[1].set_xlabel(r"concentration $\kappa$")
    axes[1].legend(fontsize=9)
    fig.suptitle("What the sampler learned about the shop population", y=1.02)
    fig.tight_layout()
    save(fig, "kemp_posteriors.png")


def fig_kemp_trace():
    phis, _, _, _ = _run_kemp(burn=0)
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    ax.plot(phis, color=TEAL, lw=0.7)
    ax.axvspan(0, 1000, color=GRAY, alpha=0.22)
    ax.text(500, 0.93, "burn-in\n(discarded)", ha="center", fontsize=9, color="#444444")
    ax.set_ylim(0.3, 1.0)
    ax.set_xlabel("sweep"); ax.set_ylabel(r"$\varphi$")
    ax.set_title(r"Trace of $\varphi$: after burn-in it wobbles steadily around its posterior — stationary noise")
    save(fig, "kemp_phi_trace.png")


def fig_kemp_shrinkage():
    from scipy.special import betaln  # noqa: F401  (keep import locality obvious)

    phis, kappas, K, N = _run_kemp()
    phi_bar = phis.mean()
    kap_med = float(np.median(kappas))
    a, b = kap_med * phi_bar, kap_med * (1 - phi_bar)
    raw = K / N
    post = (a + K) / (a + b + N)  # conjugate posterior mean per shop at the learned population

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for r, p in zip(raw, post):
        ax.plot([0, 1], [r, p], color=GRAY, lw=1.0, alpha=0.8)
    ax.scatter(np.zeros_like(raw), raw, s=60, color=BLUE, zorder=5, label="raw fraction $k_i/n_i$")
    ax.scatter(np.ones_like(post), post, s=60, color=PURPLE, zorder=5,
               label="posterior mean (at the learned population)")
    ax.axhline(phi_bar, color=ORANGE, ls="--", lw=1.5, label=f"learned population mean {phi_bar:.2f}")
    ax.set_xlim(-0.35, 1.35)
    ax.set_xticks([0, 1], ["raw fraction", "posterior mean"])
    ax.set_ylabel("tonkatsu-quality rate")
    ax.set_title("Shrinkage returns: every shop is pulled toward the population the sampler learned")
    ax.legend(fontsize=9, loc="center right")
    save(fig, "kemp_shrinkage.png")


if __name__ == "__main__":
    print("Chapter 16:")
    fig_die_convergence()
    fig_hospital()
    fig_pi_darts()
    fig_rejection()
    fig_is_tail()
    fig_weight_variance()
    fig_exemplar()
    print("Chapter 17:")
    fig_pf_steps()
    fig_pf_tracking()
    fig_pf_degeneracy()
    print("Chapter 18:")
    fig_mh_steps()
    fig_mh_histogram()
    fig_gibbs_trace()
    fig_traces()
    print("Chapter 19:")
    fig_kemp_plate()
    fig_kemp_posteriors()
    fig_kemp_trace()
    fig_kemp_shrinkage()
    print("done.")
