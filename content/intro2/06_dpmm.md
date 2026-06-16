+++
date = "2026-06-16"
title = "Dirichlet Process Mixture Models"
weight = 6
+++

## The Problem with Fixed K

In Chapter 5, we solved Chibany's bento mystery using a Gaussian Mixture Model (GMM) with K=2 components. But we had to **specify K in advance** and use BIC to validate our choice.

What if:
- We don't know how many types exist?
- The number of types changes over time?
- We want the model to discover the number of clusters automatically?

**Enter the Dirichlet Process Mixture Model (DPMM)**: A Bayesian nonparametric approach that learns the number of components from the data.

---

## The Intuition: Infinite Clusters

Imagine Chibany's supplier keeps adding new bento types over time. With a fixed-K GMM, they'd have to:
1. Notice a new type appeared
2. Re-run model selection (BIC) to choose new K
3. Refit the entire model

With a DPMM, the model **automatically** discovers new clusters as data arrives, without needing to specify K upfront.

**Key insight**: The DPMM places a prior over an **infinite** number of potential clusters, but only a finite number will actually be "active" (have observations assigned to them).

---

## The Chinese Restaurant Process Analogy

The most intuitive way to understand the DPMM is through the **Chinese Restaurant Process (CRP)**.

### The Setup

Imagine a restaurant with infinitely many tables (each table represents a cluster). Customers (observations) enter one by one and choose where to sit:

**Rule**: Customer n+1 sits:
- At an **occupied table k** with probability proportional to the number of customers already there: $\frac{n_k}{n + \alpha}$
- At a **new table** with probability: $\frac{\alpha}{n + \alpha}$

Where:
- nₖ = number of customers at table k
- α = "concentration parameter" (controls tendency to create new tables)
- n = total customers so far

### The Rich Get Richer

This creates a **rich-get-richer** dynamic:
- Popular tables attract more customers (clustering)
- But there's always a chance of starting a new table (flexibility)
- α controls the trade-off: larger α → more new tables

### Connecting to Bentos

- **Customer** = bento observation
- **Table** = cluster (bento type)
- **Seating choice** = cluster assignment
- α = how likely new bento types appear

---

## The Math: Stick-Breaking Construction

The DPMM uses a **stick-breaking** construction to define mixing proportions for infinitely many components.

### The Process

Imagine a stick of length 1. We break it into pieces:

**For k = 1, 2, 3, ..., ∞:**
1. Sample βₖ ~ Beta(1, α)
2. Set πₖ = βₖ × (1 - π₁ - π₂ - ... - πₖ₋₁)

**In plain English**:
- β₁ = fraction of stick we take for component 1
- Remaining stick: 1 - β₁
- β₂ = fraction of remaining stick we take for component 2
- π₂ = β₂ × (1 - π₁)
- And so on...

**Result**: π₁, π₂, π₃, ... sum to 1 (they're valid mixing proportions), with later components getting exponentially smaller shares.

### The Beta Distribution

βₖ ~ Beta(1, α) determines how much of the remaining stick we take:

- **α large** (e.g., α=10): Breaks are more even → many components with similar weights
- **α small** (e.g., α=0.5): First few breaks take most of the stick → few dominant components

---

## DPMM for Gaussian Mixtures: The Full Model

### Model Specification

**Stick-breaking (infinite components)**:
- For k = 1, 2, ..., K_max:
  - βₖ ~ Beta(1, α)
  - π₁ = β₁
  - πₖ = βₖ × (1 - Σⱼ₌₁ᵏ⁻¹ πⱼ) for k > 1

**Component parameters**:
- μₖ ~ N(μ₀, σ₀²) [prior on means]

**Observations** (using stick-breaking weights directly):
- For i = 1, ..., N:
  - zᵢ ~ Categorical(π) [cluster assignment using stick-breaking weights]
  - xᵢ ~ N(μ_zᵢ, σₓ²) [observation from assigned cluster]

**Important**: We use the stick-breaking weights π directly for cluster assignment. Adding an extra Dirichlet draw would create "double randomization" that makes inference much slower and less accurate!

### Why K_max?

In practice, we truncate the infinite model at some large K_max (e.g., 10 or 20). As long as K_max > the true number of clusters, this approximation is accurate.

---

## Implementing DPMM in GenJAX

Let's implement the DPMM for Chibany's bentos using the corrected approach:

<!-- validate: skip-output -->
```python
import jax
import jax.numpy as jnp
from genjax import gen, beta, normal, categorical, Target, ChoiceMap
import jax.random as random

# Hyperparameters
ALPHA = 2.0      # Concentration parameter
MU0 = 0.0        # Prior mean for cluster means
SIG0 = 4.0       # Prior std dev for cluster means
SIGX = 0.05      # Observation std dev (tight clusters)
KMAX = 10        # Maximum number of components

def make_dpmm_model(K, N):
    """
    Factory function creates DPMM model with fixed K and N

    This avoids TracerIntegerConversionError by making K and N
    closures rather than traced parameters.

    Args:
        K: Maximum number of clusters (truncation level)
        N: Number of observations
    """
    @gen
    def dpmm_model(alpha, mu0, sig0, sigx):
        """
        Dirichlet Process Mixture Model with Gaussian components

        Args:
            alpha: Concentration parameter
            mu0: Prior mean for cluster means
            sig0: Prior std dev for cluster means
            sigx: Observation std dev
        """
        # Step 1: Stick-breaking construction
        betas = []
        for k in range(K):
            beta_k = beta(1.0, alpha) @ f"beta_{k}"
            betas.append(beta_k)

        # Convert betas to pis (mixing weights)
        pis = []
        remaining = 1.0
        for k in range(K):
            pi_k = betas[k] * remaining
            pis.append(pi_k)
            remaining *= (1.0 - betas[k])

        pis_array = jnp.array(pis)
        pis_array = jnp.maximum(pis_array, 1e-6)  # Numerical stability
        pis_array = pis_array / jnp.sum(pis_array)  # Normalize

        # Step 2: Sample cluster means
        mus = []
        for k in range(K):
            mu_k = normal(mu0, sig0) @ f"mu_{k}"
            mus.append(mu_k)
        mus_array = jnp.array(mus)

        # Step 3: Generate observations
        # IMPORTANT: Use pis directly (no extra Dirichlet draw!)
        zs = []
        xs = []
        for i in range(N):
            # Cluster assignment using stick-breaking weights directly
            z_i = categorical(pis_array) @ f"z_{i}"
            zs.append(z_i)

            # Observation from assigned cluster
            x_i = normal(mus_array[z_i], sigx) @ f"x_{i}"
            xs.append(x_i)

        return {
            'mus': mus_array,
            'pis': pis_array,
            'zs': jnp.array(zs),
            'xs': jnp.array(xs),
            'betas': jnp.array(betas)
        }

    return dpmm_model

# Example: Generate synthetic data from DPMM
key = random.PRNGKey(42)

# Create model with K=10 clusters, N=20 observations
model = make_dpmm_model(K=10, N=20)

# Simulate (using default hyperparameters)
trace = model.simulate(key, (ALPHA, MU0, SIG0, SIGX))
result = trace.get_retval()

print(f"Generated data: {result['xs']}")
print(f"Cluster assignments: {result['zs']}")
print(f"Active mixing weights: {result['pis'][result['pis'] > 0.01]}")
```

**Output:**
```
Generated data: [-10.4  -9.9 -10.1   0.1   9.9  10.2 ...]
Cluster assignments: [0, 0, 0, 5, 3, 3, 3, ...]
```

Notice: The model automatically discovered active clusters (0, 3, 5 in this run), ignoring the others!

---

## Inference: A Slice Sampler for the DPMM

Now let's condition on Chibany's actual bento weights and **infer** the clusters. This is harder than the forward direction, and the choice of inference algorithm matters a lot.

### Why not plain importance sampling?

The tempting first idea is to sample whole DPMMs from the prior and keep the ones that match the data (importance/rejection sampling). It **fails badly here**: a random 10-component stick-breaking draw almost never places its means near three tight clusters at $-10, 0, +10$, so essentially every sample gets a vanishingly small weight. We need an algorithm that *moves toward* the data instead of guessing blindly.

### The slice-sampling idea

The classic solution is the **slice sampler** of Walker (2007). Its trick is to introduce one auxiliary "slice" variable per observation:

$$u_i \sim \text{Uniform}(0,\ \pi_{z_i})$$

where $\pi_{z_i}$ is the mixing weight of the cluster $i$ currently belongs to, and $\text{Uniform}(a,b)$ is the uniform distribution on the interval $[a,b]$.

Why is this useful? Given the slice values, a component $k$ is only a **candidate** for observation $i$ if its weight clears the slice, $\pi_k > u_i$. Because the stick-breaking weights shrink geometrically, only **finitely many** components ever clear the slice — so even though the model has infinitely many potential clusters, each sweep only has to consider a finite, *adaptive* set. The number of active clusters $K$ can grow or shrink from sweep to sweep as the data demand, which is exactly the behavior a nonparametric model should have. (We still allocate a generous truncation `KMAX` as a storage bound, but the slice — not the truncation — decides how many clusters are live.)

### The Gibbs sweep

Each sweep cycles through four conditional updates, sampling each quantity given the current value of the others:

1. **Slice variables** $u_i \sim \text{Uniform}(0, \pi_{z_i})$ — set the per-observation thresholds.
2. **Assignments** $z_i$ — pick a cluster from those allowed by the slice, weighted by how well it explains $x_i$: $ P(z_i = k) \propto \mathbb{1}[\pi_k > u_i]  \mathcal{N}(x_i \mid \mu_k, \sigma_x)$, where $\mathbb{1}[\cdot]$ is the indicator (1 if true, 0 if false).
3. **Stick weights** $\beta_k \sim \text{Beta}(1 + n_k,\ \alpha + \sum_{j>k} n_j)$, where $n_k$ is the number of observations now in cluster $k$ — the standard stick-breaking posterior.
4. **Cluster means** $\mu_k$ — a conjugate Normal–Normal update from the points assigned to cluster $k$ (empty clusters fall back to the prior).

We keep the explicit `for`-loops over sweeps so each step is readable; a later chapter shows how to vectorize with `scan`.

<!-- validate: tol=0.6 -->
```python
import jax
import jax.numpy as jnp
import jax.random as random
from functools import partial

# Observed bento weights (three clear clusters)
observed_weights = jnp.array([
    -10.4, -10.0, -9.4, -10.1, -9.9,   # cluster around -10
    0.0,                                 # cluster around 0
    9.5, 9.9, 10.0, 10.1, 10.5,          # cluster around +10
])
N = observed_weights.shape[0]

# Hyperparameters
ALPHA = 1.0    # concentration parameter
MU0   = 0.0    # prior mean for cluster means
SIG0  = 10.0   # prior std for cluster means
SIGX  = 1.0    # observation noise std
KMAX  = 20     # truncation / storage bound (the slice decides how many are live)

def stick_break(betas):
    """betas (K,) in (0,1) -> mixing weights pis (K,): pi_k = beta_k * prod_{j<k}(1-beta_j)."""
    log1m = jnp.log1p(-betas)
    cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(log1m)[:-1]])
    return betas * jnp.exp(cum)

def normal_logpdf(x, mu, sig):
    return -0.5 * jnp.log(2 * jnp.pi * sig**2) - 0.5 * ((x - mu) / sig)**2

def sample_betas(key, z, alpha, K):
    """Stick-breaking posterior: beta_k ~ Beta(1 + n_k, alpha + sum_{j>k} n_j)."""
    counts = jnp.bincount(z, length=K)                 # n_k
    after = jnp.cumsum(counts[::-1])[::-1]
    after = jnp.concatenate([after[1:], jnp.zeros(1)])  # sum_{j>k} n_j
    keys = random.split(key, K)
    betas = jax.vmap(lambda k, a, b: jax.random.beta(k, a, b))(keys, 1.0 + counts, alpha + after)
    return jnp.clip(betas, 1e-6, 1 - 1e-6)

def sample_mus(key, x, z, K, mu0, sig0, sigx):
    """Conjugate Normal-Normal posterior for each cluster mean (empty -> prior)."""
    counts = jnp.bincount(z, length=K)
    sums = jnp.zeros(K).at[z].add(x)
    prec0, precx = 1.0 / sig0**2, 1.0 / sigx**2
    post_prec = prec0 + counts * precx
    post_mean = (prec0 * mu0 + precx * sums) / post_prec
    post_std = jnp.sqrt(1.0 / post_prec)
    keys = random.split(key, K)
    eps = jax.vmap(lambda k: jax.random.normal(k))(keys)
    return post_mean + post_std * eps

@partial(jax.jit, static_argnums=(2,))
def gibbs_sweep(key, state, K, x, alpha, mu0, sig0, sigx):
    z, betas, mus = state
    k1, k2, k3, k4 = random.split(key, 4)
    pis = stick_break(betas)

    # 1. slice variables u_i ~ Uniform(0, pi_{z_i})
    u = jax.random.uniform(k1, (x.shape[0],)) * pis[z]

    # 2. assignments: P(z_i=k) propto 1[pi_k > u_i] * N(x_i | mu_k, sigx)
    loglik = normal_logpdf(x[:, None], mus[None, :], sigx)       # (N, K)
    logp = jnp.where(pis[None, :] > u[:, None], loglik, -jnp.inf)  # slice indicator
    keys = random.split(k2, x.shape[0])
    z = jax.vmap(lambda k, lp: jax.random.categorical(k, lp))(keys, logp)

    # 3. stick weights, 4. cluster means
    betas = sample_betas(k3, z, alpha, K)
    mus = sample_mus(k4, x, z, K, mu0, sig0, sigx)
    return (z, betas, mus)

# Run the sampler
key = random.PRNGKey(0)
z = jnp.zeros(N, dtype=jnp.int32)                # init: everyone in cluster 0
key, kb, km = random.split(key, 3)
betas = jnp.clip(jax.random.beta(kb, 1.0, ALPHA, (KMAX,)), 1e-6, 1 - 1e-6)
mus = MU0 + SIG0 * jax.random.normal(km, (KMAX,))
state = (z, betas, mus)

n_sweeps, burn = 300, 100
z_history = []
for t in range(n_sweeps):
    key, sk = random.split(key)
    state = gibbs_sweep(sk, state, KMAX, observed_weights, ALPHA, MU0, SIG0, SIGX)
    if t >= burn:
        z_history.append(state[0])

z_history = jnp.stack(z_history)                 # (n_samples, N)
z_final, betas_final, mus_final = state

# Report: relabel active clusters left-to-right by their mean for readability
active = jnp.unique(z_final)
order = sorted(active.tolist(), key=lambda k: float(mus_final[k]))
print("=== DPMM slice sampler (300 sweeps, 100 burn-in, seed 0) ===")
print(f"Discovered {len(active)} active clusters")
for rank, k in enumerate(order):
    n_k = int(jnp.sum(z_final == k))
    print(f"  Cluster {rank}: mu = {float(mus_final[k]):6.2f}   (n = {n_k})")

# Posterior over the number of occupied clusters
Ks = jnp.array([jnp.unique(z).shape[0] for z in z_history])
vals, counts = jnp.unique(Ks, return_counts=True)
print("\nPosterior over number of clusters K:")
for v, c in zip(vals, counts):
    print(f"  P(K = {int(v)}) = {float(c) / Ks.shape[0]:.2f}")
```

**Output:**
```
=== DPMM slice sampler (300 sweeps, 100 burn-in, seed 0) ===
Discovered 3 active clusters
  Cluster 0: mu = -10.03   (n = 5)
  Cluster 1: mu =   0.29   (n = 1)
  Cluster 2: mu =  10.25   (n = 5)

Posterior over number of clusters K:
  P(K = 3) = 0.58
  P(K = 4) = 0.35
  P(K = 5) = 0.07
```

The sampler **recovers all three clusters** — the five $\approx -10$ bentos, the lone $\approx 0$ bento, and the five $\approx +10$ bentos — and learns their means accurately. The posterior over $K$ also reflects genuine *uncertainty* in the number of clusters: $K=3$ is most probable, but the model gives real weight to a spurious fourth or fifth cluster — something a fixed-$K$ GMM cannot express at all.

{{% notice style="warning" title="A caveat: the posterior over $K$ is a treacherous object" %}}
It is tempting to read "$P(K = 3) = 0.58$" as the model's calibrated belief about *how many clusters really exist*. Be careful — **the marginal posterior over the number of clusters is a deep and nuanced object, and for the DPMM it does not behave the way you might hope.**

[Miller & Harrison (2014)](https://www.jmlr.org/papers/v15/miller14a.html) proved that the DPMM's posterior on the number of clusters is **inconsistent**: even when the data truly come from a finite mixture with a fixed number of components, as you collect more and more data the marginal posterior over $K$ *keeps spawning extra clusters and never settles* on the right number. Strikingly, this happens even while the model does **density estimation perfectly well** — the predictive distribution is fine, the joint is well-estimated; it is *specifically the count $K$* that misbehaves. So a DPMM is an excellent density estimator and a treacherous cluster-counter.

The good news is that this is fixable, and the fix is one careful practitioners often reach for anyway. [Ascolani, Lijoi, Rebaudo & Zanella (2022)](https://projecteuclid.org/journals/bayesian-analysis/volume-18/issue-4/Clustering-Consistency-with-Dirichlet-Process-Mixtures/10.1214/22-BA1357.full) showed that putting a **prior on the concentration parameter $\alpha$** — rather than fixing it, as we did with `ALPHA = 1.0` above — *recovers* consistency for the number of clusters when the data are generated from a finite mixture. Letting $\alpha$ itself be learned (the same "hyperprior on the prior" move you'll see in [Chapter 12](../12_hierarchical_bayes/)) is exactly the elegant remedy. The practical upshot: trust the DPMM's *predictive* fit and its *clustering* of the data, but treat any single number for "how many clusters" with suspicion unless you've put a prior on $\alpha$.
{{% /notice %}}

{{% notice style="note" title="Slice values do the truncation" %}}
We allocated `KMAX = 20` storage slots, but never assumed 20 clusters: in any sweep, only the components whose weight clears some observation's slice ($\pi_k > u_i$) are live. The data, through the slice, decide how many clusters exist — which is the whole point of going nonparametric.
{{% /notice %}}

---

## Analyzing the Posterior

The sampler gives us a *collection* of clusterings (one per post-burn-in sweep), not a single answer. Summarizing them takes a little care because of **label switching**: the cluster we call "0" in one sweep might be called "2" in the next, since the labels are arbitrary. So we cannot just average `mu_0` across sweeps — that average mixes different physical clusters together and is meaningless.

Two summaries that *are* meaningful:

**(1) A single representative clustering** — take the final sweep and relabel its clusters left-to-right by their mean, so the numbering is interpretable:

<!-- validate: tol=0.6 -->
```python
# Relabel the final sweep's clusters 0..K-1 by increasing mean
active = jnp.unique(z_final)
order = sorted(active.tolist(), key=lambda k: float(mus_final[k]))
relabel = {k: r for r, k in enumerate(order)}
mode_assignments = jnp.array([relabel[int(z)] for z in z_final])

print("Cluster assignment per bento:", [int(v) for v in mode_assignments])
print("\nCluster means:")
for r, k in enumerate(order):
    n_k = int(jnp.sum(z_final == k))
    print(f"  Cluster {r}: μ = {float(mus_final[k]):6.2f}   (n = {n_k})")
```

**Output:**
```
Cluster assignment per bento: [0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2]

Cluster means:
  Cluster 0: μ = -10.03   (n = 5)
  Cluster 1: μ =   0.29   (n = 1)
  Cluster 2: μ =  10.25   (n = 5)
```

**(2) A label-invariant summary** — the **co-clustering probability** that two bentos land in the *same* cluster, averaged over all samples. This sidesteps label switching entirely, because "same cluster?" doesn't depend on what the cluster is named:

<!-- validate: tol=0.15 -->
```python
# P(bento i and bento j share a cluster), averaged over posterior samples
same_cluster = jnp.mean(
    (z_history[:, :, None] == z_history[:, None, :]).astype(jnp.float32),
    axis=0,
)

print("Co-clustering probability matrix P(i ~ j):")
for row in jnp.round(same_cluster, 2):
    print("  [" + " ".join(f"{float(v):.2f}" for v in row) + "]")
```

**Output:**
```
Co-clustering probability matrix P(i ~ j):
  [1.00 0.92 0.90 0.88 0.90 0.00 0.00 0.00 0.00 0.00 0.00]
  [0.92 1.00 0.88 0.87 0.93 0.00 0.00 0.00 0.00 0.00 0.00]
  [0.90 0.88 1.00 0.88 0.88 0.00 0.00 0.00 0.00 0.00 0.00]
  [0.88 0.87 0.88 1.00 0.90 0.00 0.00 0.00 0.00 0.00 0.00]
  [0.90 0.93 0.88 0.90 1.00 0.00 0.00 0.00 0.00 0.00 0.00]
  [0.00 0.00 0.00 0.00 0.00 1.00 0.00 0.00 0.00 0.00 0.00]
  [0.00 0.00 0.00 0.00 0.00 0.00 1.00 0.90 0.93 0.90 0.86]
  [0.00 0.00 0.00 0.00 0.00 0.00 0.90 1.00 0.88 0.86 0.90]
  [0.00 0.00 0.00 0.00 0.00 0.00 0.93 0.88 1.00 0.88 0.90]
  [0.00 0.00 0.00 0.00 0.00 0.00 0.90 0.86 0.88 1.00 0.86]
  [0.00 0.00 0.00 0.00 0.00 0.00 0.86 0.90 0.90 0.86 1.00]
```

The block structure is unmistakable: the five $\approx -10$ bentos (rows 0–4) almost always share a cluster with each other and **never** with the rest; the lone $\approx 0$ bento (row 5) sits by itself; the five $\approx +10$ bentos (rows 6–10) form the third block. The model recovered the three groups **without ever being told there were three** — and the within-block probabilities sitting a little below 1.0 honestly reflect the small chance, seen in the posterior over $K$, that a group occasionally splits.

---

## A Trap: Label Switching

We sidestepped a subtle but important problem above, and it's worth making explicit because it bites *every* mixture model, not just the DPMM.

**The cluster labels are arbitrary.** Nothing in the model distinguishes "cluster 0" from "cluster 2" — the likelihood
$$p(x \mid z, \mu) = \prod_i \mathcal{N}(x_i \mid \mu_{z_i}, \sigma_x)$$
is **completely unchanged** if we swap the names of two clusters and swap their means to match. The model has a built-in symmetry: with $K$ occupied clusters there are $K!$ equivalent labelings of the *same* clustering, all with identical posterior probability.

**Why this breaks naive summaries.** A good sampler will, over many sweeps, wander between these equivalent labelings — the group sitting at $-10$ might be called cluster 0 in one sweep and cluster 2 in the next. So if you compute a per-label average like
$$\bar\mu_0 = \frac{1}{S}\sum_{s} \mu_0^{(s)},$$
you are averaging the $-10$ group's mean in some sweeps with the $+10$ group's mean in others. The result is mush — typically a number near the overall data mean with a huge standard deviation, which *looks* like a failed inference even when the sampler worked perfectly. (Try it: averaging `mu_0` across our sweeps gives something like $\mu \approx 0 \pm 9$, which is nonsense — the sampler is fine; the *summary* is wrong.)

**The fixes** — all of which we used or could use here:

1. **Report label-invariant quantities.** The co-clustering matrix above never asks "what is cluster $k$?", only "are $i$ and $j$ together?", so label switching simply cannot affect it. This is the most robust option and the one to reach for first. The posterior over the *number* of clusters $K$ is label-invariant too.
2. **Summarize a single representative sample**, not an average across samples — e.g. the final sweep (or the highest-posterior sweep), relabeled into a canonical order. That's what `mode_assignments` did: we sorted the clusters left-to-right by mean so "cluster 0" always denotes the lightest group.
3. **Impose an identifiability constraint / relabel post hoc.** Pin an ordering (e.g. $\mu_0 < \mu_1 < \mu_2$) or run a relabeling algorithm (Stephens, 2000) that permutes each sweep's labels to best match a reference before averaging. Then per-label averages become meaningful again.

{{% notice style="warning" title="Don't average raw per-label parameters" %}}
If you find yourself writing `jnp.mean(mu_k for each sweep)` over MCMC samples of a mixture model, stop. Either summarize a *label-invariant* function of the clustering, or relabel the samples into a canonical order first. Raw per-label averages silently conflate different clusters and make a healthy sampler look broken.
{{% /notice %}}

---

## The Posterior Predictive Distribution

**Question**: What weight should Chibany expect for the next bento?

To predict the next bento's weight, we draw from the recovered mixture: pick a cluster in proportion to how many bentos it holds, then sample a weight from that cluster's Gaussian. We use the representative (final-sweep) clustering for a clean, interpretable predictive.

<!-- validate: tol=1.0 -->
```python
# Mixing weights from the representative clustering: proportion of bentos per cluster
counts = jnp.bincount(z_final, length=KMAX)
weights = counts / counts.sum()                  # zero for empty clusters

def draw_one(k):
    k1, k2 = random.split(k)
    z_new = jax.random.categorical(k1, jnp.log(weights + 1e-12))   # pick a cluster
    return mus_final[z_new] + SIGX * jax.random.normal(k2)         # sample its Gaussian

key, sk = random.split(key)
predictions = jax.vmap(draw_one)(random.split(sk, 5000))

print(f"Posterior predictive mean: {float(jnp.mean(predictions)):.2f}")
print(f"Posterior predictive std:  {float(jnp.std(predictions)):.2f}")
for label, lo, hi in [("≈ -10", -15, -5), ("≈  0", -5, 5), ("≈ +10", 5, 15)]:
    frac = float(jnp.mean((predictions >= lo) & (predictions < hi)))
    print(f"  P(next bento {label}) = {frac:.2f}")
```

**Output:**
```
Posterior predictive mean: 0.21
Posterior predictive std:  9.72
  P(next bento ≈ -10) = 0.45
  P(next bento ≈  0) = 0.09
  P(next bento ≈ +10) = 0.46
```

The posterior predictive is **multimodal** — a mixture of the three clusters — so its overall mean ($\approx 0$) is *not* a sensible prediction: no bento actually weighs around zero. The useful statement is the per-mode breakdown: the next bento is about equally likely to be a light ($\approx -10$) or heavy ($\approx +10$) type, with a small chance of the rare middle type. Let's visualize it!

---

## Visualizing the Results

```python
import matplotlib.pyplot as plt
from scipy.stats import norm as scipy_norm
```

<details>
<summary>Click to show visualization code</summary>

```python
import jax.numpy as jnp

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Representative clustering (final sweep), relabeled left-to-right by mean
active = jnp.unique(z_final)
order = sorted(active.tolist(), key=lambda k: float(mus_final[k]))
colors = ['red', 'green', 'blue', 'purple', 'orange']

# Left: observed data colored by recovered cluster, with cluster centers
ax1.scatter(observed_weights, jnp.zeros_like(observed_weights),
            s=120, alpha=0.4, color='gray', label='Observed data')
for rank, k in enumerate(order):
    mu = float(mus_final[k])
    members = observed_weights[z_final == k]
    color = colors[rank % len(colors)]
    ax1.scatter(members, jnp.zeros_like(members) + 0.05, s=120, color=color)
    ax1.axvline(mu, color=color, linestyle='--', alpha=0.7,
                label=f'Cluster {rank}: μ={mu:.1f}')

ax1.set_xlabel('Weight')
ax1.set_yticks([])
ax1.set_title('Recovered Clusters')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: posterior predictive distribution with each cluster's contribution
ax2.hist(predictions, bins=50, density=True, alpha=0.5, color='gray',
         edgecolor='black', label='Posterior predictive')

counts = jnp.bincount(z_final, length=KMAX)
weights = counts / counts.sum()
x_range = jnp.linspace(-15, 15, 1000)
for rank, k in enumerate(order):
    mu = float(mus_final[k])
    w = float(weights[k])
    color = colors[rank % len(colors)]
    cluster_pdf = w * scipy_norm.pdf(x_range, mu, SIGX)
    ax2.plot(x_range, cluster_pdf, color=color, linewidth=2,
             label=f'Cluster {rank} (π≈{w:.2f})')

ax2.set_xlabel('Weight')
ax2.set_ylabel('Density')
ax2.set_title('Posterior Predictive Distribution')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dpmm_results.png', dpi=150, bbox_inches='tight')
plt.show()
```

</details>

![DPMM: Discovered 3 Clusters](../../images/intro2/dpmm_results.png)

**The visualization shows**:
- **Left**: Observed data points with posterior cluster centers and uncertainties
- **Right**: Trimodal posterior predictive (mixture of three Gaussians)

---

## Comparing DPMM to Fixed-K GMM

| Feature | Fixed-K GMM | DPMM |
|---------|-------------|------|
| **K specified?** | Yes (must choose K) | No (learned from data) |
| **Model selection** | BIC, cross-validation | Automatic |
| **New clusters** | Requires refitting | Discovered automatically |
| **Computational cost** | Lower (fixed K) | Higher (infinite K, truncated) |
| **Uncertainty in K** | Not modeled | Naturally captured |

**When to use DPMM**:
- Unknown number of clusters
- Exploratory data analysis
- Data arrives sequentially (online learning)
- Want Bayesian uncertainty quantification

**When to use Fixed-K GMM**:
- K is known or strongly constrained
- Computational efficiency matters
- Simpler implementation preferred

---

## The Role of α (Concentration Parameter)

α controls the tendency to create new clusters:

```python
# Try different alpha values
alphas = [0.1, 1.0, 5.0, 20.0]
```

<details>
<summary>Click to show visualization code</summary>

```python
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for ax, alpha in zip(axes, alphas):
    # Generate stick-breaking weights
    key = random.PRNGKey(42)
    betas = []
    pis = []

    for k in range(20):  # Show first 20 components
        key, subkey = random.split(key)
        beta_k = jax.random.beta(subkey, 1.0, alpha)
        betas.append(beta_k)

        if k == 0:
            pi_k = beta_k
        else:
            pi_k = beta_k * (1.0 - sum(pis))
        pis.append(pi_k)

    # Plot
    ax.bar(range(20), pis)
    ax.set_xlabel('Component')
    ax.set_ylabel('Mixing Proportion')
    ax.set_title(f'α = {alpha}')
    ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('stick_breaking_alpha.png', dpi=150, bbox_inches='tight')
plt.show()
```

</details>

![Stick-Breaking Process with Different α Values](../../images/intro2/stick_breaking_alpha.png)

**Interpretation**:
- **α = 0.1**: First component dominates (few clusters)
- **α = 1.0**: Moderate spread (balanced)
- **α = 5.0**: More components active (many clusters)
- **α = 20.0**: Very even spread (diffuse)

---

## Real-World Applications

### Anomaly Detection
- Normal data forms clusters
- Outliers create singleton clusters
- α controls sensitivity to outliers

### Topic Modeling
- Documents are mixtures over topics
- DPMM discovers number of topics automatically
- Each topic is a distribution over words

### Genomics
- Cluster genes by expression patterns
- Number of functional groups unknown
- DPMM identifies distinct expression profiles

### Image Segmentation
- Pixels cluster by color/texture
- DPMM finds natural segments
- No need to specify number of segments

---

## Practice Problems

### Problem 1: Adjusting α

Using the observed bento data from earlier, run inference with α ∈ {0.5, 2.0, 10.0}.

**a)** How does the number of active clusters change?

**b)** How does posterior uncertainty change?

<details>
<summary>Show Solution</summary>

We reuse the `gibbs_sweep` from earlier, just rerunning the sampler at each $\alpha$ and reporting the **average number of occupied clusters** (averaging over sweeps, which avoids the single-sweep noise):

<!-- validate: tol=0.5 -->
```python
def run_sampler(alpha, seed=0, n_sweeps=300, burn=100):
    key = random.PRNGKey(seed)
    z = jnp.zeros(N, dtype=jnp.int32)
    key, kb, km = random.split(key, 3)
    betas = jnp.clip(jax.random.beta(kb, 1.0, alpha, (KMAX,)), 1e-6, 1 - 1e-6)
    mus = MU0 + SIG0 * jax.random.normal(km, (KMAX,))
    state = (z, betas, mus)
    z_hist = []
    for t in range(n_sweeps):
        key, sk = random.split(key)
        state = gibbs_sweep(sk, state, KMAX, observed_weights, alpha, MU0, SIG0, SIGX)
        if t >= burn:
            z_hist.append(state[0])
    return jnp.stack(z_hist)

for alpha in [0.5, 2.0, 10.0]:
    z_hist = run_sampler(alpha)
    Ks = jnp.array([jnp.unique(z).shape[0] for z in z_hist])
    print(f"α = {alpha:4.1f}:  E[K] = {float(jnp.mean(Ks)):.2f}")
```

**Output:**
```
α =  0.5:  E[K] = 3.21
α =  2.0:  E[K] = 3.60
α = 10.0:  E[K] = 4.76
```

The trend is exactly as the theory predicts: a larger concentration parameter $\alpha$ makes the model spin up **more** clusters (some of them spurious splits of the three real groups), while a small $\alpha$ keeps it parsimonious. Note that even at $\alpha = 0.5$ the model still finds the three genuine clusters — the data are clearly separated enough that the likelihood overrides the prior's pull toward fewer clusters.
</details>

---

### Problem 2: Sequential Learning

Chibany receives bentos one at a time. Implement **online learning** where the model updates as each bento arrives.

**Hint**: One simple approach reuses the slice sampler you already have — after each new bento arrives, rerun the sampler on *all data seen so far* and report the occupied clusters. (A more efficient approach would *warm-start* from the previous posterior instead of restarting; that is the idea behind sequential Monte Carlo.)

<details>
<summary>Show Solution (sketch)</summary>

This is left as an implementation exercise. The structure below is **pseudo-code** — `run_sampler` is the function from Problem 1; the point is the outer loop over a growing data prefix, not a new inference algorithm:

<!-- validate: skip -->
```python
def online_dpmm(data_stream):
    """Rerun the slice sampler on a growing prefix of the data."""
    for i in range(1, len(data_stream) + 1):
        prefix = data_stream[:i]                 # all bentos seen so far
        z_hist = run_sampler_on(prefix)          # adapt run_sampler to take the data
        K = average_num_clusters(z_hist)         # E[K] over sweeps, as in Problem 1
        print(f"After {i} bentos: E[K] ≈ {K:.1f}")

online_dpmm(observed_weights)
```

**Expected behavior**: the number of occupied clusters grows as genuinely new bento types first appear, then stabilizes once each type has been seen — the model commits to a new cluster only when the data force it to.
</details>

---

## What We've Accomplished

We started with a mystery: bentos with an average weight that doesn't match any individual bento. Through this tutorial, we:

1. **Chapter 1**: Understood expected value paradoxes in mixtures
2. **Chapter 2**: Learned continuous probability (PDFs, CDFs)
3. **Chapter 3**: Mastered the Gaussian distribution
4. **Chapter 4**: Performed Bayesian learning for parameters
5. **Chapter 5**: Built Gaussian Mixture Models with EM
6. **Chapter 6**: Extended to infinite mixtures with DPMM

**You now have the tools to**:
- Model complex, multimodal data
- Discover latent structure automatically
- Quantify uncertainty in clustering
- Perform Bayesian inference with GenJAX

### Where this goes next

Clustering was about finding structure *in a pile of data*. The chapters ahead turn the same Bayesian machinery toward new questions:

- **[Chapter 7: Bayesian Generalization](../07_generalization/)** asks how you learn a *concept* from a handful of examples — the same posterior-over-hypotheses idea, but now the hypotheses are *sets* (rules), and the payoff is a model of how humans generalize.
- **[Chapters 8–11: the Bayesian-networks spine](../08_bayes_nets/)** zoom out from a single model to the *structure* of models: drawing them as graphs (Bayes nets), reading off which variables inform which (conditional independence and d-separation), distinguishing *seeing* from *doing* (causal Bayes nets and the do-operator), and measuring it all in bits (information theory). The DPMM you just built is itself a Bayes net — Chapter 8 makes that explicit.
- **[Chapter 12: Hierarchical Bayes](../12_hierarchical_bayes/)** stacks priors on priors so the model can *learn its own prior* from related problems — and, as we noted above, it's exactly the move that tames the DPMM's cluster-count behavior.

So the mystery bentos were just the beginning: the rest of Tutorial 3 is about graphs, causes, information, and learning the prior itself.

---

## Further Reading

### Theoretical Foundations
- Ferguson (1973): "A Bayesian Analysis of Some Nonparametric Problems" (original DP paper)
- Teh et al. (2006): "Hierarchical Dirichlet Processes" (extensions to HDP)
- Austerweil, Gershman, Tenenbaum, & Griffiths (2015): "Structure and Flexibility in Bayesian Models of Cognition" (Chapter in The Oxford Handbook of Computational and Mathematical Psychology - comprehensive overview of Bayesian nonparametric approaches to cognitive modeling)

### Practical Implementations
- Neal (2000): "Markov Chain Sampling Methods for Dirichlet Process Mixture Models" (MCMC inference)
- Walker (2007): "Sampling the Dirichlet Mixture Model with Slices" (the slice sampler used in this chapter)
- Kalli, Griffin, & Walker (2011): "Slice sampling mixture models" (refinements and a clear exposition)
- Blei & Jordan (2006): "Variational Inference for Dirichlet Process Mixtures" (scalable inference)
- Stephens (2000): "Dealing with label switching in mixture models" (post-hoc relabeling for valid per-component summaries)

### GenJAX Documentation
- [GenJAX GitHub](https://github.com/probcomp/genjax) - Official repository
- [Probabilistic Programming Examples](https://www.gen.dev/) - Gen.jl (sister project)

---

{{% notice style="tip" title="Key Takeaways" %}}
1. **DPMM**: Bayesian nonparametric model that learns K automatically
2. **Stick-breaking**: Defines mixing proportions for infinite components
3. **CRP**: Intuitive "customers and tables" interpretation
4. **α**: Concentration parameter controlling cluster tendency
5. **Slice sampler**: Auxiliary slice variables $u_i$ adaptively truncate the infinite stick, so each Gibbs sweep only handles finitely many live clusters
6. **Label switching**: Cluster labels are arbitrary — summarize with label-invariant quantities (co-clustering, posterior over $K$) or a single relabeled sample, never raw per-label averages
{{% /notice %}}

---

## Interactive Exploration

Want to experiment with DPMMs yourself? Try our **interactive Jupyter notebook** that lets you:

- Adjust the concentration parameter α and see its effect on clustering
- Add or remove data points and watch the model adapt
- Change the truncation level K_max
- Visualize posterior distributions in real-time

{{% notice style="success" title="Try It Yourself!" %}}
**📓 [Open Interactive DPMM Notebook on Google Colab](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/dpmm_interactive.ipynb)**

No installation required - runs directly in your browser!
{{% /notice %}}

The notebook includes:
- Complete DPMM implementation with stick-breaking
- Interactive widgets for all parameters
- Real-time visualization of posteriors
- Guided exercises to deepen understanding

This is a great way to build intuition for how α, K_max, and the data itself interact to produce the posterior distribution.

---

## References

- Ascolani, F., Lijoi, A., Rebaudo, G., & Zanella, G. (2023). Clustering consistency with Dirichlet process mixtures. *Biometrika, 110*(2), 551–558. <https://doi.org/10.1093/biomet/asac051>
- Austerweil, J. L., Gershman, S. J., Tenenbaum, J. B., & Griffiths, T. L. (2015). Structure and flexibility in Bayesian models of cognition. In J. R. Busemeyer, Z. Wang, J. T. Townsend, & A. Eidels (Eds.), *The Oxford handbook of computational and mathematical psychology* (pp. 187–208). Oxford University Press.
- Blei, D. M., & Jordan, M. I. (2006). Variational inference for Dirichlet process mixtures. *Bayesian Analysis, 1*(1), 121–143. <https://doi.org/10.1214/06-BA104>
- Ferguson, T. S. (1973). A Bayesian analysis of some nonparametric problems. *The Annals of Statistics, 1*(2), 209–230. <https://doi.org/10.1214/aos/1176342360>
- Kalli, M., Griffin, J. E., & Walker, S. G. (2011). Slice sampling mixture models. *Statistics and Computing, 21*(1), 93–105. <https://doi.org/10.1007/s11222-009-9150-y>
- Miller, J. W., & Harrison, M. T. (2014). Inconsistency of Pitman–Yor process mixtures for the number of components. *Journal of Machine Learning Research, 15*(96), 3333–3370. <https://jmlr.org/papers/v15/miller14a.html>
- Neal, R. M. (2000). Markov chain sampling methods for Dirichlet process mixture models. *Journal of Computational and Graphical Statistics, 9*(2), 249–265. <https://doi.org/10.1080/10618600.2000.10474879>
- Stephens, M. (2000). Dealing with label switching in mixture models. *Journal of the Royal Statistical Society: Series B (Statistical Methodology), 62*(4), 795–809. <https://doi.org/10.1111/1467-9868.00265>
- Teh, Y. W., Jordan, M. I., Beal, M. J., & Blei, D. M. (2006). Hierarchical Dirichlet processes. *Journal of the American Statistical Association, 101*(476), 1566–1581. <https://doi.org/10.1198/016214506000000302>
- Walker, S. G. (2007). Sampling the Dirichlet mixture model with slices. *Communications in Statistics — Simulation and Computation, 36*(1), 45–54. <https://doi.org/10.1080/03610910601096262>
