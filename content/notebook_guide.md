+++
date = "2026-06-03"
title = "Interactive Notebooks - All Tutorials"
weight = 99
+++

## Interactive Jupyter Notebooks

This page provides a comprehensive overview of all Jupyter notebooks available across the tutorial series. Each notebook opens directly in Google Colab for immediate interactive exploration.

**How to use these notebooks:**
- 📓 Click "Open in Colab" to launch the notebook in your browser
- ✏️ Run cells, modify code, and experiment with parameters
- 💾 Save a copy to your Google Drive to keep your changes
- 📚 Return to the linked tutorial chapters for detailed explanations

---

## Tutorial 1: Discrete Probability

### First GenJAX Model
**Notebook**: [📓 Open in Colab: `first_model.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/first_model.ipynb)

**What it covers:**
- Your first probabilistic model in GenJAX
- Simulating Chibany's lunch choices (hamburger vs tonkatsu)
- Basic probability calculations with discrete outcomes
- Understanding random sampling and probability distributions

**Related Tutorial Chapters:**
- [Tutorial 1, Chapter 3: Probability from Counting](../intro/03_prob_count/)

**Topics:**
- Discrete probability distributions
- Random sampling
- GenJAX basics
- Probability visualization

---

### Conditioning and Bayes' Rule
**Notebook**: [📓 Open in Colab: `conditioning.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/conditioning.ipynb)

**What it covers:**
- Conditional probability in practice
- Implementing Bayes' rule in GenJAX
- The taxicab problem with interactive examples
- Sequential belief updating

**Related Tutorial Chapters:**
- [Tutorial 1, Chapter 4: Conditional Probability](../intro/04_conditional/)

**Topics:**
- Conditional probability
- Bayes' theorem
- Posterior belief updating
- Prior and likelihood

---

### Bayesian Learning
**Notebook**: [📓 Open in Colab: `bayesian_learning.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/bayesian_learning.ipynb)

**What it covers:**
- Complete taxicab problem with visualizations
- Sequential Bayesian updating with multiple observations
- Interactive sliders to explore different base rates and accuracies
- How prior beliefs affect posterior conclusions

**Related Tutorial Chapters:**
- [Tutorial 1, Chapter 5: Bayes' Theorem](../intro/05_bayes/)
- [Tutorial 2 (GenJAX), Chapter 4: Conditioning](../genjax/04_conditioning/)

**Topics:**
- Bayesian inference
- Sequential updating
- Base rate effects
- Prior-posterior relationships

---

## Tutorial 2: GenJAX Programming

### Your First GenJAX Model (Tutorial 2)
**Notebook**: [📓 Open in Colab: `02_first_genjax_model.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/02_first_genjax_model.ipynb)

**What it covers:**
- Building generative models in GenJAX
- Interactive widgets to adjust parameters in real-time
- Visualizing probability distributions
- Understanding how parameter changes affect outcomes

**Related Tutorial Chapters:**
- [Tutorial 2 (GenJAX), Chapter 0: Getting Started](../genjax/00_getting_started/)
- [Tutorial 2 (GenJAX), Chapter 2: First Model](../genjax/02_first_model/)

**Topics:**
- GenJAX generative functions
- Parameter exploration
- Interactive visualization
- Model simulation

---

## Tutorial 3: Continuous Probability & Bayesian Learning

### Gaussian Bayesian Interactive Exploration
**Notebook**: [📓 Open in Colab: `gaussian_bayesian_interactive_exploration.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/gaussian_bayesian_interactive_exploration.ipynb)

**What it covers:**
- **Part 1: Gaussian-Gaussian Bayesian Updates**
  - Interactive sliders to adjust likelihood variance
  - Sequential observation addition with real-time posterior updates
  - Comparison of posterior vs predictive distributions
  - Effect of measurement noise on learning

- **Part 2: Gaussian Mixture Categorization**
  - How priors affect decision boundaries
  - Effect of variance ratios on categorization
  - Marginal (mixture) distribution visualization
  - Understanding bimodal vs unimodal distributions

**Related Tutorial Chapters:**
- [Tutorial 3 (Intro2), Main Page](../intro2/)
- [Tutorial 3 (Intro2), Chapter 4: Bayesian Learning](../intro2/04_bayesian_learning/)
- [Tutorial 3 (Intro2), Chapter 5: Mixture Models](../intro2/05_mixture_models/)

**Topics:**
- Gaussian distributions
- Bayesian parameter learning
- Conjugate priors
- Posterior inference
- Mixture models
- Decision boundaries

---

### Assignment 1: Gaussian Bayesian Update
**Notebook**: [📓 Open in Colab: `solution_1_gaussian_bayesian_update.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/solution_1_gaussian_bayesian_update.ipynb)

**What it covers:**
- **Part (a)**: Visualizing the prior distribution
- **Part (b)**: Effect of likelihood variance (σ²_x = 0.25 vs. 4)
- **Part (c)**: Effect of number of observations (N=1 vs. N=5)
- **Part (d)**: Precision-weighted averaging in action
- **Part (e)**: Posterior vs. predictive distributions
- **Part (f)**: GenJAX verification of analytical formulas

**Related Tutorial Chapters:**
- [Tutorial 3 (Intro2), Chapter 4: Bayesian Learning](../intro2/04_bayesian_learning/)

**Topics:**
- Gaussian conjugate priors
- Likelihood variance effects
- Posterior concentration
- Predictive distributions
- Precision weighting

---

### Assignment 2: Gaussian Clusters
**Notebook**: [📓 Open in Colab: `solution_2_gaussian_clusters.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/solution_2_gaussian_clusters.ipynb)

**What it covers:**
- **Part (a)**: Deriving P(category|observation) using Bayes' rule
- **Part (b)**: Effect of priors on decision boundaries
- **Part (c)**: Effect of variance ratios on categorization
- **Part (d)**: Computing marginal distributions
- **Part (e)**: Understanding bimodal vs. unimodal mixtures
- **Part (f)**: GenJAX simulation of mixture models

**Related Tutorial Chapters:**
- [Tutorial 3 (Intro2), Chapter 4: Bayesian Learning - Problem 2](../intro2/04_bayesian_learning/#problem-2-gaussian-clusters-preview-of-chapter-5)
- [Tutorial 3 (Intro2), Chapter 5: Mixture Models](../intro2/05_mixture_models/)

**Topics:**
- Mixture model categorization
- Bayes' rule with continuous distributions
- Decision boundaries
- Marginal probability
- Mixture distributions

---

### Dirichlet Process Mixture Models (DPMM)
**Notebook**: [📓 Open in Colab: `dpmm_interactive.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/dpmm_interactive.ipynb)

**What it covers:**
- Interactive DPMM exploration
- Automatic cluster discovery
- Chinese Restaurant Process visualization
- Infinite mixture models
- Bayesian nonparametrics

**Related Tutorial Chapters:**
- [Tutorial 3 (Intro2), Chapter 6: DPMM](../intro2/06_dpmm/)

**Topics:**
- Dirichlet Process
- Infinite mixture models
- Chinese Restaurant Process
- Bayesian nonparametrics
- Automatic model selection

---

### Bayesian Generalization
**Notebook**: [📓 Open in Colab: `07_generalization.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/07_generalization.ipynb)

**What it covers:**
- The sticker warm-up: a concept as a *set* of hypotheses
- The number game over 1–30 with seven candidate rules
- The **size principle** under weak vs. strong sampling
- The generalization gradient — predicting which new numbers fit

**Related Tutorial Chapters:**
- [Tutorial 3 (Intro2), Chapter 7: Bayesian Generalization](../intro2/07_generalization/)

**Topics:**
- Hypotheses as sets
- The size principle
- Weak vs. strong sampling
- Generalization gradients

---

### Bayesian Networks
**Notebook**: [📓 Open in Colab: `08_bayes_nets.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/08_bayes_nets.ipynb)

**What it covers:**
- The Chapter 5 mixture model, re-built explicitly as a Bayes net
- A hierarchical version (a mixing-weight prior stacked on top)
- Chibany's multi-parent bento network with a conditional probability table
- Inference from an observed weight back to the hidden cluster

**Related Tutorial Chapters:**
- [Tutorial 3 (Intro2), Chapter 8: Bayesian Networks](../intro2/08_bayes_nets/)

**Topics:**
- Directed acyclic graphs (DAGs)
- The Markov factorization
- Conditional probability tables
- Ancestral sampling and inference

---

### Conditional Independence & d-Separation
**Notebook**: [📓 Open in Colab: `09_conditional_independence.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/09_conditional_independence.ipynb)

**What it covers:**
- The rain / sprinkler / wet-floor collider as a runnable model
- Conditioning on evidence and recovering posteriors by importance sampling
- Watching **explaining away** happen numerically (0.30 → 0.59 → 0.30)

**Related Tutorial Chapters:**
- [Tutorial 3 (Intro2), Chapter 9: Conditional Independence and d-Separation](../intro2/09_conditional_independence/)

**Topics:**
- Chain, fork, and collider patterns
- d-separation
- The Markov blanket
- Explaining away

---

### Causal Bayes Nets & the Do-Operator
**Notebook**: [📓 Open in Colab: `10_causal_bayes_nets.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/10_causal_bayes_nets.ipynb)

**What it covers:**
- The smoking / teeth / cancer network as observational vs. interventional models
- Computing P(cancer | teeth) vs. P(cancer | do(teeth)) by Monte Carlo
- Seeing the see/do gap (≈0.098 vs. 0.052) emerge from graph surgery

**Related Tutorial Chapters:**
- [Tutorial 3 (Intro2), Chapter 10: Causal Bayes Nets and the Do-Operator](../intro2/10_causal_bayes_nets/)

**Topics:**
- The do-operator and graph surgery
- Confounders
- Observational vs. interventional distributions
- Pearl's ladder of causation

---

### Information Theory
**Notebook**: [📓 Open in Colab: `11_information_theory.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/11_information_theory.ipynb)

**What it covers:**
- Estimating entropy and mutual information by Monte Carlo
- The collider creating mutual information from nothing
- I(rain; tea) = 0 jumping to I(rain; tea | sign) ≈ 0.46 bits — explaining away, in bits

**Related Tutorial Chapters:**
- [Tutorial 3 (Intro2), Chapter 11: Information Theory](../intro2/11_information_theory/)

**Topics:**
- Surprise and entropy
- Mutual information
- Independence in information units
- The collider, in bits

---

### Markov Chains
**Notebook**: [📓 Open in Colab: `13_markov_chains.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/13_markov_chains.ipynb)

**What it covers:**
- Chibany's bento chain as a transition matrix; sampling sequences from it
- Power iteration converging to the 70/30 stationary distribution from any start
- The stationary distribution as the eigenvalue-1 eigenvector
- A three-state worked example

**Related Tutorial Chapters:**
- [Tutorial 3 (Intro2), Chapter 13: Markov Chains](../intro2/13_markov_chains/)

**Topics:**
- The Markov property and transition matrices
- Stationary distributions and power iteration
- Ergodicity
- GenJAX sequence sampling + `jax.lax.scan`

---

### Random Walks on Networks
**Notebook**: [📓 Open in Colab: `14_random_walks_networks.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/14_random_walks_networks.ipynb)

**What it covers:**
- Chibany's animal network as an adjacency matrix, row-normalized to a transition matrix
- The stationary distribution of a random walk: π ∝ degree (Cat the bridge wins)
- Visit frequency by simulation, matching the degree law
- Hand-rolled PageRank with the ε-teleport on a tiny directed web

**Related Tutorial Chapters:**
- [Tutorial 3 (Intro2), Chapter 14: Random Walks on Networks](../intro2/14_random_walks_networks/)

**Topics:**
- Graphs, adjacency matrices, degree
- Random walk as a Markov chain on nodes
- π ∝ degree and where it breaks (directed graphs)
- PageRank

---

### Memory Search
**Notebook**: [📓 Open in Colab: `15_memory_search.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/15_memory_search.ipynb)

**What it covers:**
- A censored random walk on a small semantic network
- The censoring function (report each animal on first visit) and inter-item response times
- The position-1-slowest "switch cost" signature, with no switch rule
- A simulation-based (ABC) sketch that recovers block structure from fluency lists

**Related Tutorial Chapters:**
- [Tutorial 3 (Intro2), Chapter 15: Memory Search as a Random Walk](../intro2/15_memory_search/)

**Topics:**
- Semantic fluency and clustering/switching
- Censoring; first-hitting times; IRT
- Recovering the human optimal-foraging curve from one process
- Inverting the walk (U-INVITE, SNAFU) and simulation-based inference

---

## Recommended Learning Paths

### Path 1: Complete Beginner
1. `first_model.ipynb` - Start here for GenJAX basics
2. `conditioning.ipynb` - Learn conditional probability
3. `bayesian_learning.ipynb` - Master Bayesian updating
4. `02_first_genjax_model.ipynb` - Build your first full model
5. `gaussian_bayesian_interactive_exploration.ipynb` - Explore continuous probability
6. `solution_1_gaussian_bayesian_update.ipynb` - Practice Gaussian inference
7. `solution_2_gaussian_clusters.ipynb` - Learn mixture models
8. `dpmm_interactive.ipynb` - Advanced: infinite mixtures

### Path 2: Bayesian Learning Focus
1. `bayesian_learning.ipynb` - Discrete Bayes' rule
2. `gaussian_bayesian_interactive_exploration.ipynb` - Continuous Bayesian inference
3. `solution_1_gaussian_bayesian_update.ipynb` - Gaussian conjugate priors
4. `solution_2_gaussian_clusters.ipynb` - Mixture model inference
5. `dpmm_interactive.ipynb` - Bayesian nonparametrics

### Path 3: Quick Interactive Tour
1. `02_first_genjax_model.ipynb` - Interactive parameter exploration
2. `gaussian_bayesian_interactive_exploration.ipynb` - Bayesian learning with sliders
3. `dpmm_interactive.ipynb` - Automatic clustering

### Path 4: Graphical Models & Causality
1. `08_bayes_nets.ipynb` - Draw models as graphs
2. `09_conditional_independence.ipynb` - d-separation and explaining away
3. `10_causal_bayes_nets.ipynb` - Seeing vs. doing (the do-operator)
4. `11_information_theory.ipynb` - Measure dependence in bits

### Path 5: Markov Chains, Networks & Memory
1. `13_markov_chains.ipynb` - Transition matrices and the stationary distribution
2. `14_random_walks_networks.ipynb` - Random walks on graphs, π ∝ degree, PageRank
3. `15_memory_search.ipynb` - Recall as a censored random walk on a semantic network

---

## Tips for Using Notebooks

**Getting Started:**
- Click "Open in Colab" to launch any notebook
- Run cells in order (Shift+Enter) to execute code
- Experiment by changing parameter values

**Interactive Widgets:**
- Many notebooks include sliders and controls
- Adjust parameters and see results update in real-time
- Try extreme values to understand edge cases

**Saving Your Work:**
- File → Save a copy in Drive (saves to your Google Drive)
- Your experiments and notes will be preserved
- You can share your modified notebooks with others

**Troubleshooting:**
- If code doesn't run, try Runtime → Restart runtime
- Make sure to run cells in order from top to bottom
- Check that all required packages are installed (usually automatic in Colab)

---

## All Notebooks at a Glance

| Notebook | Tutorial | Topics | Difficulty |
|----------|----------|--------|------------|
| `first_model.ipynb` | Tutorial 1 | Discrete probability, basics | ⭐ Beginner |
| `conditioning.ipynb` | Tutorial 1 | Conditional probability | ⭐ Beginner |
| `bayesian_learning.ipynb` | Tutorials 1 & 2 | Bayesian inference | ⭐⭐ Intermediate |
| `02_first_genjax_model.ipynb` | Tutorial 2 | GenJAX programming | ⭐⭐ Intermediate |
| `gaussian_bayesian_interactive_exploration.ipynb` | Tutorial 3 | Continuous Bayes, mixtures | ⭐⭐⭐ Advanced |
| `solution_1_gaussian_bayesian_update.ipynb` | Tutorial 3 | Gaussian inference | ⭐⭐⭐ Advanced |
| `solution_2_gaussian_clusters.ipynb` | Tutorial 3 | Mixture models | ⭐⭐⭐ Advanced |
| `dpmm_interactive.ipynb` | Tutorial 3 | Bayesian nonparametrics | ⭐⭐⭐⭐ Expert |
| `07_generalization.ipynb` | Tutorial 3 | Concept learning, size principle | ⭐⭐⭐ Advanced |
| `08_bayes_nets.ipynb` | Tutorial 3 | Bayesian networks, DAGs | ⭐⭐⭐ Advanced |
| `09_conditional_independence.ipynb` | Tutorial 3 | d-separation, explaining away | ⭐⭐⭐ Advanced |
| `10_causal_bayes_nets.ipynb` | Tutorial 3 | Causal inference, do-operator | ⭐⭐⭐ Advanced |
| `11_information_theory.ipynb` | Tutorial 3 | Entropy, mutual information | ⭐⭐⭐ Advanced |
| `13_markov_chains.ipynb` | Tutorial 3 | Markov chains, stationary distribution | ⭐⭐⭐ Advanced |
| `14_random_walks_networks.ipynb` | Tutorial 3 | Random walks, PageRank, π ∝ degree | ⭐⭐⭐ Advanced |
| `15_memory_search.ipynb` | Tutorial 3 | Censored walk, memory fluency | ⭐⭐⭐ Advanced |

---

**Need help?** Return to the [main tutorial page](../) or consult the [glossary](../glossary/) for term definitions.

**Enjoying the notebooks?** This tutorial series is generously funded by the [Japanese Probabilistic Computing Consortium Association (JPCCA)](https://jpcca.org/).
