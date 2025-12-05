+++
date = "2025-12-05"
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

---

**Need help?** Return to the [main tutorial page](../) or consult the [glossary](../glossary/) for term definitions.

**Enjoying the notebooks?** This tutorial series is generously funded by the [Japanese Probabilistic Computing Consortium Association (JPCCA)](https://jpcca.org/).
