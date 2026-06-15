+++
date = "2026-06-15"
title = "Continuous Probability and Bayesian Learning"
# NOTE: Ch7 (Bayesian Generalization) is a page bundle: content/intro2/07_generalization/ with
#   _index.md (overview) + 4 part pages. The mermaid node below is a diagram label, not a link.
weight = 3
toc = true
+++

## From Discrete to Continuous with GenJAX

In the first two tutorials, you learned:
1. **Tutorial 1**: Probability theory using discrete outcomes (sets and counting)
2. **Tutorial 2**: How to express probability in code using GenJAX

Now you'll learn to work with **continuous probability distributions** and perform **Bayesian learning** on real-valued data, all using the GenJAX tools you've already learned!

![Chibany with scale](images/chibanylayingdown.png)

## The New Challenge

In Tutorial 1, Chibany's lunch choices were **discrete**: tonkatsu OR hamburger. But what if we want to model **continuous measurements** like:

- The **weight** of their bento box
- The **temperature** of their office
- The **time** it takes students to arrive

These aren't discrete choices. They're **continuous values** that can fall anywhere on a number line!

## Learning Path

Here's your journey into continuous probability and Bayesian learning:

```mermaid
graph TB
    A[1. Mystery Bentos] --> B[2. Continuous Probability]
    B --> C[3. Gaussian Distribution]
    C --> D[4. Bayesian Learning]
    D --> E[5. Mixture Models]
    E --> F[6. Dirichlet Process]
    F --> G[7. Bayesian Generalization]
    G --> H[8. Bayesian Networks]
    H --> I[9. Conditional Independence]
    I --> J[10. Causal Bayes Nets]
    J --> K[11. Information Theory]
    K --> L[12. Hierarchical Bayes]
    L --> M[13. Markov Chains]
    M --> N[14. Random Walks on Networks]
    N --> O[15. Memory Search]
    O --> P[16. Monte Carlo]
    P --> Q[17. Particle Filtering]
    Q --> R[18. MCMC]
    R --> S[19. Sampling the Mind]

    style B fill:#27ae60
    style D fill:#27ae60
    style E fill:#27ae60
    style G fill:#27ae60
    style L fill:#27ae60

    classDef foundational fill:#27ae60,stroke:#333,stroke-width:2px,color:#fff
```

**Foundational Chapters** (green): Core continuous probability concepts—PDFs, Bayesian updating, and mixture models.

**Prerequisites**: Complete Tutorial 1 (Probability) and Tutorial 2 (GenJAX) before starting here.

## What You'll Learn

This tutorial builds directly on GenJAX (Tutorial 2) to explore:

### Chapter 1: Chibany's Mystery Bentos (Expected Value)
- The paradox of averages in mixtures
- Expected value and balance points
- Why averages can be misleading
- **GenJAX**: Simulating mixture distributions

### Chapter 2: Continuous Random Variables
- Probability density functions (PDFs)
- Cumulative distribution functions (CDFs)
- The uniform distribution
- **GenJAX**: Sampling from and conditioning on continuous distributions

### Chapter 3: The Gaussian Distribution
- The bell curve and its properties
- Mean and variance parameters
- The 68-95-99.7 rule
- **GenJAX**: Working with Normal distributions

### Chapter 4: Bayesian Learning with Gaussians
- Prior beliefs about parameters
- Updating beliefs with data (conjugate priors)
- Posterior and predictive distributions
- **GenJAX**: Implementing Gaussian-Gaussian models
- **📓 Interactive Assignments**: Hands-on exploration of parameter effects

### Chapter 5: Gaussian Mixture Models
- Combining multiple distributions
- Clustering with mixtures
- The complete bento model
- **GenJAX**: Building and inferring mixture models

### Chapter 6: Dirichlet Process Mixture Models
- Infinite mixture models
- The Dirichlet Process prior
- Automatic model selection
- **GenJAX**: Implementing DPMM for clustering

### Chapter 7: Bayesian Generalization
- Concepts as hypotheses: a hypothesis is a *set*
- The posterior-weighted vote; weak vs. strong sampling and the size principle
- Shepard's law of generalization, *emerging from the model* rather than assumed
- No Free Lunch: why a learner needs a prior
- **GenJAX**: Enumerating hypothesis spaces and computing generalization gradients

### Chapter 8: Bayesian Networks
- The mixture model from Chapter 5, re-seen as a graph: nodes, arrows, and parents
- The Markov factorization — reading a joint distribution straight off a DAG
- Multi-parent networks and conditional probability tables; the parameter-counting argument
- **GenJAX**: building Bayes nets as generative functions, ancestral sampling, and inference from effect back to cause

### Chapter 9: Conditional Independence and d-Separation
- The three building blocks: chain, fork, and collider — and which way each passes information
- Why conditioning on a collider *opens* a path (the Monty Hall surprise)
- The d-separation algorithm and the Markov blanket
- **Explaining away**, walked through numerically; **GenJAX**: the rain/sprinkler collider

### Chapter 10: Causal Bayes Nets and the Do-Operator
- When arrows mean *causes*: three causal stories behind one correlation
- Intervention as graph surgery; the do-operator $do(X = x)$
- $P(Y \mid X)$ vs. $P(Y \mid do(X))$ — seeing vs. doing, with worked numbers
- Pearl's ladder of causation; the blicket detector; **GenJAX**: observational vs. interventional models

### Chapter 11: Information Theory — Surprise, Uncertainty, and the Collider
- Surprise as $-\log P(x)$; entropy as expected surprise; the bit
- Mutual information — how many bits one variable carries about another
- Independence and d-separation, restated in information units
- The collider creating mutual information from nothing (explaining away, in bits); KL divergence as a forward pointer; **GenJAX**: entropy and MI by Monte Carlo

### Chapter 12: Hierarchical Bayes
- Two extremes that fail: no pooling vs. complete pooling
- The Beta distribution and Beta-Binomial conjugacy
- Partial pooling and shrinkage: borrowing strength across groups
- Learning the prior itself — inference one level up (the answer to No Free Lunch)
- **GenJAX**: A hierarchical generative model + importance sampling over hyperparameters

### Chapter 13: Markov Chains — The Future Forgets the Past
- The Markov property: the future depends on the past only through the present
- Transition matrices (row-stochastic) and the two views — state diagram vs. matrix
- The stationary distribution $\pi$, found by power iteration or as the eigenvalue-1 eigenvector
- **GenJAX**: sampling a sequence from the matrix; long-run frequency by simulation

### Chapter 14: Random Walks on Networks
- Graphs $G = (V, E)$; turning an adjacency matrix into a transition matrix by row-normalizing
- A random walk as a Markov chain whose states are nodes; the law $\pi_i \propto \deg(i)$ for undirected walks
- PageRank as the stationary distribution of a teleporting random surfer (and *Google and the Mind*)
- **GenJAX**: sampling a walk on Chibany's animal network; hand-rolled PageRank on a tiny directed web

### Chapter 15: Memory Search as a Random Walk
- Semantic fluency: people recall in bursts-by-category — and recall is a random walk on a semantic network (Abbott, Austerweil & Griffiths 2012)
- The censoring function: report each animal only on its first visit; first-hitting times $\tau(k)$ and the inter-item response time
- One memoryless process reproduces the "switch-cost" signature with no switch rule — a simpler account than optimal foraging
- **GenJAX**: a censored random walk that reproduces the position-1-slowest IRT curve; forward pointer to MCMC

### Chapter 16: Monte Carlo — Estimating by Sampling
- Estimating an expectation (and any probability, as the mean of an indicator) by sampling and averaging; the $1/\sqrt{n}$ error rate
- Rejection sampling and inverse-CDF; importance sampling — draw from an easy proposal $q$, reweight by $w = p/q$
- Self-normalized importance sampling for unnormalized posteriors; likelihood weighting; the effective sample size
- **GenJAX**: importance sampling with `model.importance` → (trace, log_weight)

### Chapter 17: Particle Filtering
- Streaming inference: *yesterday's posterior is today's prior*; the state-space model (motion + observation)
- The particle filter loop — weight, resample, propagate — and weight degeneracy as the reason resampling is needed
- A worked 1-D tracking example; particle filters as a process model of human inference (limited memory, order effects)
- **GenJAX**: the propagate step as a `@gen` motion model inside the filter loop

### Chapter 18: Markov Chain Monte Carlo
- Running Chapter 13 backwards: *design* a Markov chain whose stationary distribution is a chosen target
- Metropolis–Hastings (why the normalizer cancels; detailed balance) and Gibbs sampling (resampling a full conditional, always accepting)
- Burn-in, mixing, and the multimodal trap — good local acceptance does not imply good global mixing
- **GenJAX**: assembling an MH step from the `assess` scoring primitive

### Chapter 19: Sampling the Mind
- MCMC with People — a person choosing between options *is* the Metropolis accept step, so the chain reveals their prior
- A hybrid Gibbs–Metropolis sampler for the hierarchical Beta-Binomial: Gibbs the per-unit rates, collapse them out, Metropolis the population
- The mean/concentration reparametrization $(\varphi, \kappa)$; reading off a predictive from the learned population
- **GenJAX**: the Gibbs step as a conjugate `beta` draw; the Metropolis step scoring the Beta-Binomial marginal

## Prerequisites

**Required:**
- ✅ Completed "A Narrative Introduction to Probability" (Tutorial 1)
- ✅ Completed "Probabilistic Programming with GenJAX" (Tutorial 2)
- ✅ Comfortable writing and running GenJAX generative functions
- ✅ Understand traces, conditioning, and inference in GenJAX

**Not Required:**
- ❌ Calculus (we provide intuition and use GenJAX for computation)
- ❌ Advanced statistics
- ❌ Mathematical proofs

## Learning Philosophy

**You already know** how to think probabilistically (Tutorial 1) and how to express probability in GenJAX code (Tutorial 2). This tutorial shows you how those same ideas extend to the continuous case!

**Key insight:** Moving from discrete to continuous isn't about learning entirely new concepts. It's about adapting what you know:
- **Probabilities** become **probability densities**
- **Sums** become **integrals** (but GenJAX handles this for you!)
- **Counting** becomes **measuring area under curves**

## What Makes This Tutorial Different

Unlike traditional probability courses, we:

1. **Use GenJAX throughout**: Every concept is illustrated with runnable code
2. **Leverage simulation**: When math gets complex, we approximate with samples
3. **Build intuition first**: Visual understanding before mathematical details
4. **Connect to Tutorial 1**: Every concept links back to discrete probability
5. **Interactive notebooks**: Adjust parameters and see results update live

## Why Continuous Probability Matters

Many real-world phenomena are naturally continuous:

- **Scientific measurements**: Temperature, weight, time, distance
- **Financial data**: Stock prices, returns, volatility
- **Machine learning**: Most input features are continuous
- **Natural phenomena**: Heights, speeds, concentrations

Understanding continuous probability lets you model the real world more accurately!

## Tutorial Structure

Each chapter includes:
- 📖 **Concept explanation** connecting to discrete probability
- 💻 **GenJAX implementation** with runnable code
- 🎮 **Interactive Colab notebooks** to explore parameters
- 📊 **Visualizations** showing PDFs, posteriors, and predictions
- ✅ **Exercises** with solutions

---

## 📓 Interactive Notebooks & Assignments

This tutorial includes **three comprehensive Jupyter notebooks** for hands-on learning:

### 1. Interactive Exploration Notebook

**File**: [📓 Open in Colab: `gaussian_bayesian_interactive_exploration.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/gaussian_bayesian_interactive_exploration.ipynb)

**What it does**: Provides interactive widgets to explore concepts in real-time
- **Part 1**: Gaussian-Gaussian Bayesian updates
  - Adjust likelihood variance and see posterior change
  - Add observations sequentially and watch learning happen
  - Compare posterior vs. predictive distributions
- **Part 2**: Gaussian mixture categorization
  - Explore how priors affect decision boundaries
  - See effect of variance ratios on categorization
  - Visualize marginal (mixture) distributions

**When to use**:
- While reading Chapter 4 (to build intuition)
- Before attempting the assignments (to see concepts visually)
- When reviewing (to refresh understanding)

### 2. Assignment Solution Notebooks

#### Problem 1: Gaussian Bayesian Update
**File**: [📓 Open in Colab: `solution_1_gaussian_bayesian_update.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/solution_1_gaussian_bayesian_update.ipynb)

**Linked from**: Chapter 4, Section "Exploration Exercise"

**Topics covered**:
- Effect of likelihood variance (σ²_x) on learning
- Effect of number of observations (N) on posterior concentration
- Precision-weighted averaging in action
- Posterior vs. predictive distributions
- GenJAX verification of analytical formulas

#### Problem 2: Gaussian Clusters
**File**: [📓 Open in Colab: `solution_2_gaussian_clusters.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/solution_2_gaussian_clusters.ipynb)

**Linked from**: Chapter 4 (Preview section) and Chapter 5 (Prerequisite section)

**Topics covered**:
- Deriving P(category|observation) using Bayes' rule
- Effect of priors on decision boundaries
- Effect of variance ratios on categorization
- Computing marginal distributions
- Understanding bimodal vs. unimodal mixtures
- GenJAX simulation of mixture models

### How to Use the Notebooks

{{% notice style="success" title="📚 Recommended Learning Path" %}}
1. **Read Chapter 4** up to "Sequential Learning" section
2. **Open the interactive exploration notebook** and experiment with parameters
3. **Work through Assignment 1** (Gaussian Bayesian Update)
4. **Continue reading Chapter 4** through the end
5. **Work through Assignment 2** (Gaussian Clusters) as preparation for Chapter 5
6. **Read Chapter 5** with confidence in your understanding of categorization!
{{% /notice %}}

**Tips**:
- Run all cells in order first, then go back and modify parameters
- Compare your intuition with the printed interpretations
- Try extreme parameter values to see edge cases
- Use the GenJAX implementations as templates for your own models

---

## Ready to Begin?

Let's start with Chibany's mystery: why does the average weight of their bentos seem impossible?

[Next: Chapter 1 - Chibany's Mystery Bentos →](./01_mystery_bentos.md)

---

## A Note on Mathematics

This tutorial includes some mathematical notation (PDFs, integrals, etc.), but **you don't need to be a math expert!**

**When you see an integral** (∫): Think "area under the curve" (GenJAX computes it for you)
**When you see a derivative**: Think "rate of change" (rarely needed, GenJAX handles it)
**When you see Σ vs ∫**: Think "sum vs continuous sum" (same idea, different notation)

**Focus on:**
- What the code does (run it and see!)
- What the plots show (visual intuition)
- How it connects to discrete probability (concepts you know)

Mathematics provides precise definitions, but GenJAX lets you **compute without deriving**!

---

{{% notice style="tip" title="Learning Tip" %}}
**Start with visuals, then code, then math.** When you see a new concept:

1. Look at the plots first (what does it look like?)
2. Run the GenJAX code (what happens?)
3. Read the math (why does it work?)

Understanding beats memorization. You can always reference the math later!
{{% /notice %}}

---

Special thanks to [JPPCA](https://jpcca.org/) for their generous support of this tutorial series.
