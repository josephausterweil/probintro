+++
date = "2026-07-04"
title = "Interactive Notebooks - All Tutorials"
weight = 910
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
- [Tutorial 1, Chapter 3: Probability from Counting](../foundations/probability-as-counting/)

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
- [Tutorial 1, Chapter 4: Conditional Probability](../foundations/conditional-probability/)

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
- [Tutorial 1, Chapter 5: Bayes' Theorem](../foundations/bayes-rule/)
- [Tutorial 2 (GenJAX), Chapter 4: Conditioning](../genjax/conditioning/)

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
- [Tutorial 2 (GenJAX), Chapter 0: Getting Started](../genjax/getting-started/)
- [Tutorial 2 (GenJAX), Chapter 2: First Model](../genjax/first-model/)

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
- [Tutorial 3 (Intro2), Main Page](../learning/)
- [Tutorial 3 (Intro2), Chapter 4: Bayesian Learning](../learning/bayesian-learning/)
- [Tutorial 3 (Intro2), Chapter 5: Mixture Models](../learning/mixture-models/)

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
- [Tutorial 3 (Intro2), Chapter 4: Bayesian Learning](../learning/bayesian-learning/)

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
- [Tutorial 3 (Intro2), Chapter 4: Bayesian Learning - Problem 2](../learning/bayesian-learning/#problem-2-gaussian-clusters-preview-of-chapter-5)
- [Tutorial 3 (Intro2), Chapter 5: Mixture Models](../learning/mixture-models/)

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
- [Tutorial 3 (Intro2), Chapter 6: DPMM](../complexity/dpmm/)

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
- [Tutorial 3 (Intro2), Chapter 7: Bayesian Generalization](../structure/generalization/)

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
- [Tutorial 3 (Intro2), Chapter 8: Bayesian Networks](../structure/bayes-nets/)

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
- [Tutorial 3 (Intro2), Chapter 9: Conditional Independence and d-Separation](../structure/conditional-independence/)

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
- [Tutorial 3 (Intro2), Chapter 10: Causal Bayes Nets and the Do-Operator](../structure/causal-bayes-nets/)

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
- [Tutorial 3 (Intro2), Chapter 11: Information Theory](../structure/information-theory/)

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
- [Tutorial 3 (Intro2), Chapter 13: Markov Chains](../sampling/markov-chains/)

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
- [Tutorial 3 (Intro2), Chapter 14: Random Walks on Networks](../sampling/random-walks-networks/)

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
- [Tutorial 3 (Intro2), Chapter 15: Memory Search as a Random Walk](../sampling/memory-search/)

**Topics:**
- Semantic fluency and clustering/switching
- Censoring; first-hitting times; IRT
- Recovering the human optimal-foraging curve from one process
- Inverting the walk (U-INVITE, SNAFU) and simulation-based inference

---

### Monte Carlo
**Notebook**: [📓 Open in Colab: `16_monte_carlo.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/16_monte_carlo.ipynb)

**What it covers:**
- The Monte Carlo estimator: averaging a die roll to 3.5, estimating π by throwing darts
- Rejection sampling and the indicator function; importance sampling with weights $w = p/q$
- Self-normalized importance sampling (unnormalized posteriors) and the effective sample size
- GenJAX importance sampling via `model.importance`

**Related Tutorial Chapters:**
- [Tutorial 3 (Intro2), Chapter 16: Monte Carlo](../sampling/monte-carlo/)

**Topics:**
- Estimating expectations and probabilities by sampling; the $1/\sqrt{n}$ rate
- Rejection sampling, inverse-CDF, importance sampling
- Effective sample size as a proposal-quality diagnostic

---

### Particle Filtering
**Notebook**: [📓 Open in Colab: `17_particle_filtering.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/17_particle_filtering.ipynb)

**What it covers:**
- Tracking Chibany down a corridor from noisy sensor pings
- The particle filter loop — weight, resample, propagate — and what each step does
- Weight degeneracy and why resampling is the cure (ESS collapsing without it)
- The GenJAX motion model as the propagate step

**Related Tutorial Chapters:**
- [Tutorial 3 (Intro2), Chapter 17: Particle Filtering](../sampling/particle-filtering/)

**Topics:**
- State-space models; sequential importance sampling
- Weight → resample → propagate; degeneracy
- Particle filters as a process model of human inference

---

### Markov Chain Monte Carlo
**Notebook**: [📓 Open in Colab: `18_markov_chain_monte_carlo.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/18_markov_chain_monte_carlo.ipynb)

**What it covers:**
- Metropolis–Hastings on a bimodal target; why the normalizer cancels
- Gibbs sampling on a correlated 2-D Gaussian (always accepts)
- Mixing, burn-in, and the multimodal trap (a trapped chain from two starts)
- Assembling an MH step in GenJAX from the `assess` scoring primitive

**Related Tutorial Chapters:**
- [Tutorial 3 (Intro2), Chapter 18: Markov Chain Monte Carlo](../sampling/mcmc/)

**Topics:**
- Designing a chain to hit a target; detailed balance
- Metropolis–Hastings and Gibbs sampling
- Burn-in, mixing, and multimodal targets

---

### Sampling the Mind
**Notebook**: [📓 Open in Colab: `19_sampling_the_mind.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/19_sampling_the_mind.ipynb)

**What it covers:**
- MCMC with People: a person's choices as the Metropolis accept step
- A hybrid Gibbs–Metropolis sampler for the hierarchical Beta-Binomial (bento shops)
- The Beta-Binomial marginal (collapsing θ out) and the mean/concentration reparametrization
- Reading off a predictive for a brand-new, unseen unit

**Related Tutorial Chapters:**
- [Tutorial 3 (Intro2), Chapter 19: Sampling the Mind](../sampling/sampling-the-mind/)

**Topics:**
- MCMC with People; recovering a prior from choices
- Hybrid Gibbs + Metropolis on a hierarchical model
- Beta-Binomial conjugacy and the collapsed sampler

---

### Statistical Decision Theory
**Notebook**: [📓 Open in Colab: `20_statistical_decision_theory.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/20_statistical_decision_theory.ipynb)

**What it covers:**
- From beliefs to actions: the decision problem, loss, and risk
- Bayes vs minimax criteria, disagreeing on the same numbers
- The loss → estimator rule: 0–1 → mode (MAP), squared → mean, absolute → median
- Expected loss by sampling the posterior in GenJAX
- *One and Done*: why optimal decisions can come from a single sample (Vul et al.)

**Related Tutorial Chapters:**
- [Tutorial 3 (Intro2), Chapter 20: Statistical Decision Theory](../decisions/decision-theory/)

**Topics:**
- Decision theory, loss functions, risk, Bayes vs minimax
- MAP / mean / median as Bayes estimators
- Monte-Carlo decisions and probability matching

---

### Markov Decision Processes
**Notebook**: [📓 Open in Colab: `21_markov_decision_processes.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/21_markov_decision_processes.ipynb)

**What it covers:**
- Building an MDP: Markov chain + reward + a choice of transition matrix
- The transition as a GenJAX generative model
- Value, the Bellman equation, and value iteration on the Chibany MDP
- The discount factor γ and the policy-flip threshold
- Estimating a value by simulating rollouts (the Monte-Carlo bridge)

**Related Tutorial Chapters:**
- [Tutorial 3 (Intro2), Chapter 21: Markov Decision Processes](../decisions/mdps/)

**Topics:**
- MDPs, policies, the Bellman equation, value iteration
- Discounting and the γ-flip
- Planning by simulation

---

### Q-Learning
**Notebook**: [📓 Open in Colab: `22_q_learning.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/22_q_learning.ipynb)

**What it covers:**
- The model-free TD update: learning rate α and the TD error
- Q-learning on the GardenPath gridworld
- Reward shaping and positive cycles; potential-based shaping as the fix
- Dyna and MCTS: planning with a learned model (simulation-based RL)
- Reward hacking → RLHF; the TD error → dopamine

**Related Tutorial Chapters:**
- [Tutorial 3 (Intro2), Chapter 22: Q-Learning](../decisions/q-learning/)

### Inverse RL: Reading Goals from Behavior
**Notebook**: [📓 Open in Colab: `23_inverse_rl_goal_inference.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/23_inverse_rl_goal_inference.ipynb)

**What it covers:**
- Goal inference as Bayes' rule with a softmax policy as the likelihood
- The GenJAX observer: posterior over goals, and the freeze-frame curve
- Ill-posedness, and how the prior + rationality β resolve it
- Theory of Mind = inverse RL (Baker & Tenenbaum)
- IRL at scale: MaxEnt, GAIL, AIRL

**Related Tutorial Chapters:**
- [Tutorial 3 (Intro2), Chapter 23: Inverse RL](../decisions/inverse-rl/)

### POMDPs and Belief: Inferring a Hidden World
**Notebook**: [📓 Open in Colab: `24_pomdps_belief_inference.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/24_pomdps_belief_inference.ipynb)

**What it covers:**
- Belief states b(s) and the Tiger belief update in GenJAX
- α-vectors and the decision threshold (open vs. listen)
- A POMDP is an MDP over beliefs
- Bayesian Theory of Mind (belief + desire)
- Teaching, legibility, and Cooperative IRL

**Related Tutorial Chapters:**
- [Tutorial 3 (Intro2), Chapter 24: POMDPs](../decisions/pomdps-belief/)

### Modern RL: Preferences, World Models, and Machine Minds
**Notebook**: [📓 Open in Colab: `25_modern_rl_world_models.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/25_modern_rl_world_models.ipynb)

**What it covers:**
- RLHF/DPO as preference-based inverse RL (Bradley–Terry reward model in GenJAX)
- Recovering a reward from pairwise preferences; the additive-constant ambiguity
- Amortized vs. Bayesian Theory of Mind (ToMnet)
- World models (MuZero, Dreamer)
- The skeptical LLM Theory-of-Mind debate

**Related Tutorial Chapters:**
- [Tutorial 3 (Intro2), Chapter 25: Modern RL](../decisions/modern-rl-world-models/)

### From Bentos to Vectors: the Linear Algebra You Need
**Notebook**: [📓 Open in Colab: `vectors_and_spaces.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/vectors_and_spaces.ipynb)

**What it covers:**
- Data → vectors: measurements as components, dimension, points in a plane
- Dot product and cosine similarity, fully worked (kind vs. size)
- 4-D vectors and embedding arithmetic (king − man + woman ≈ queen)
- Matrix–vector multiplication as "a dot product per row"; columns = where the basis arrows land
- Composition (two machines collapse into one) and a first ReLU layer

**Related Tutorial Chapters:**
- [Part VIII, From Bentos to Vectors](../deep/vectors-and-spaces/)

### Capstone — From Rules to Weights
**Notebook**: [📓 Open in Colab: `capstone_from_rules_to_weights.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/capstone_from_rules_to_weights.ipynb)

**What it covers:**
- A three-way race on the bento data as data grows: ridge on hand features, a 2-4-1 learned-feature net, and RBF kernel ridge
- Fixed features win when data is scarce; learned/kernel methods overtake as it grows
- Optional stretch: the Belkin et al. (2019) double-descent curve

**Related Tutorial Chapters:**
- [Part VIII, From Rules to Weights](../deep/from-rules-to-weights/)

### Capstone — Neural Network Fundamentals
**Notebook**: [📓 Open in Colab: `capstone_neural_net_fundamentals.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/capstone_neural_net_fundamentals.ipynb)

**What it covers:**
- Part A: replicate Minsky & Papert — a linear unit fails XOR, a hidden layer solves it
- Part B: the same 2-layer net on a real task (sklearn digits, 0 vs. 1)
- Anchor: Minsky & Papert (1969) + Rumelhart, Hinton & Williams (1986)

**Related Tutorial Chapters:**
- [Part VIII, Neural Network Fundamentals](../deep/neural-net-fundamentals/)

### Capstone — Transformers & Attention
**Notebook**: [📓 Open in Colab: `capstone_transformers_attention.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/capstone_transformers_attention.ipynb)

**What it covers:**
- Build single-head attention from scratch; hand-set keys/queries for a "look at the previous token" head
- Then a mini induction pattern; visualize the weight matrices as heatmaps
- Anchor: Vaswani et al. (2017) + Olsson et al. (2022) induction heads

**Related Tutorial Chapters:**
- [Part VIII, Transformers & Attention](../deep/transformers-attention/)

### Capstone — LLMs & In-Context Learning (the flagship)
**Notebook**: [📓 Open in Colab: `capstone_llms_in_context_learning.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/capstone_llms_in_context_learning.ipynb)

**What it covers:**
- Part A: the exact Bayesian posterior-predictive of a latent-mixture model vs. a genuinely-trained predictor — watch the posterior sharpen with examples (Xie-style)
- Part B: implement the Falck-style martingale test and run it on both — the exact Bayesian passes; is your trained model's gap noise or systematic drift?
- Part C (stretch): sketch the same test on a real LLM
- **Runs both sides of the ICL-is-Bayesian debate**

**Related Tutorial Chapters:**
- [Part VIII, LLMs & In-Context Learning](../deep/llms-in-context-learning/)

### Capstone — World Models & Imagination
**Notebook**: [📓 Open in Colab: `capstone_world_models_imagination.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/capstone_world_models_imagination.ipynb)

**What it covers:**
- Part A: fit a dynamics model on a gridworld from random-policy data
- Part B: plan by imagination (random-shooting rollouts) and race it against model-free Q-learning on sample efficiency
- Part C (stretch): inject model error and watch planning degrade
- Anchor: Dyna (Sutton 1991) + MuZero (Schrittwieser 2020)

**Related Tutorial Chapters:**
- [Part VIII, World Models & Imagination](../deep/world-models-imagination/)

### Capstone — Adversarial Examples
**Notebook**: [📓 Open in Colab: `capstone_adversarial_examples.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/capstone_adversarial_examples.ipynb)

**What it covers:**
- Part A: implement the fast gradient sign method (FGSM) and flip a confident classifier
- Part B: sweep the perturbation size ε and plot the robustness curve
- Part C (stretch): a pass of adversarial training, then re-measure
- Anchor: Goodfellow, Shlens & Szegedy (2015) + Szegedy et al. (2014)

**Related Tutorial Chapters:**
- [Part IX, Adversarial Examples](../ethics/adversarial-examples/)

### Capstone — Fairness, Formally
**Notebook**: [📓 Open in Colab: `capstone_fairness_formalisms.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/capstone_fairness_formalisms.ipynb)

**What it covers:**
- Part A: implement demographic parity, equalized odds, and calibration on a confusion matrix
- Part B: reproduce the impossibility result on a COMPAS-like dataset with unequal base rates
- Part C (stretch): the fairness frontier (the tradeoff curve)
- Anchor: Kleinberg et al. (2016) + Chouldechova (2017) + the ProPublica/COMPAS case

**Related Tutorial Chapters:**
- [Part IX, Fairness, Formally](../ethics/fairness-formalisms/)

### Capstone — Bias in Data
**Notebook**: [📓 Open in Colab: `capstone_bias_in_data.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/capstone_bias_in_data.ipynb)

**What it covers:**
- Part A: implement WEAT on a toy embedding and measure the effect size
- Part B: attempt linear debiasing and measure what's removed vs. what leaks back
- Part C (stretch): a downstream task inheriting the bias
- Anchor: Caliskan et al. (2017) + Bolukbasi et al. (2016) + Buolamwini & Gebru (2018)

**Related Tutorial Chapters:**
- [Part IX, Bias in Data](../ethics/bias-in-data/)

### Capstone — Alignment & Safety (the book's last)
**Notebook**: [📓 Open in Colab: `capstone_alignment_safety.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/capstone_alignment_safety.ipynb)

**What it covers:**
- Part A: fit a Bradley–Terry reward model from pairwise preferences (recover a latent reward)
- Part B: demonstrate reward hacking — optimize against the learned reward and watch it exploit a gap
- Part C (stretch): more diverse preferences shrink the ill-posedness
- Anchor: Christiano et al. (2017) + Hadfield-Menell et al. (2016) + the Goodhart/specification-gaming literature

**Related Tutorial Chapters:**
- [Part IX, Alignment & Safety](../ethics/alignment-safety/)

**Topics:**
- Q-learning, TD error, ε-greedy exploration
- Reward shaping, positive cycles, reward hacking
- Dyna, Monte Carlo Tree Search, simulation-based RL

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

### Path 6: Sampling & Monte Carlo
1. `16_monte_carlo.ipynb` - Estimating by sampling; importance sampling and effective sample size
2. `17_particle_filtering.ipynb` - Streaming inference; weight → resample → propagate
3. `18_markov_chain_monte_carlo.ipynb` - Designing a chain to hit a target; Metropolis–Hastings and Gibbs
4. `19_sampling_the_mind.ipynb` - MCMC with People and a sampler for the hierarchical Beta-Binomial

### Path 7: Decisions & Reinforcement Learning
1. `20_statistical_decision_theory.ipynb` - From beliefs to actions: loss, risk, Bayes vs minimax, and one-and-done decisions
2. `21_markov_decision_processes.ipynb` - MDPs, the Bellman equation, value iteration, and planning by simulation
3. `22_q_learning.ipynb` - Model-free Q-learning, reward shaping and positive cycles, Dyna and MCTS
4. `23_inverse_rl_goal_inference.ipynb` - Inverse RL: goal inference, softmax likelihood, ill-posedness, ToM = IRL
5. `24_pomdps_belief_inference.ipynb` - POMDPs, belief updates, α-vectors, Bayesian ToM, teaching & legibility
6. `25_modern_rl_world_models.ipynb` - RLHF/DPO as inverse RL, world models, the LLM theory-of-mind debate
7. `vectors_and_spaces.ipynb` - Linear algebra from zero: vectors, dot products, matrices as transformations (the Part VIII on-ramp)
8. `capstone_from_rules_to_weights.ipynb` - Capstone: ridge vs learned features vs kernel ridge as data grows
9. `capstone_neural_net_fundamentals.ipynb` - Capstone: XOR from Minsky & Papert to a real digit classifier
10. `capstone_transformers_attention.ipynb` - Capstone: build attention from scratch; previous-token and induction heads
11. `capstone_llms_in_context_learning.ipynb` - Capstone: run both sides of the ICL-is-Bayesian debate (Xie/Ye vs Falck)
12. `capstone_world_models_imagination.ipynb` - Capstone: learn a world model, plan by imagination vs model-free Q-learning
13. `capstone_adversarial_examples.ipynb` - Capstone: implement FGSM, flip a classifier, plot the robustness curve
14. `capstone_fairness_formalisms.ipynb` - Capstone: the three fairness criteria + the impossibility result as code
15. `capstone_bias_in_data.ipynb` - Capstone: WEAT on embeddings, linear debiasing and its leaks
16. `capstone_alignment_safety.ipynb` - Capstone: fit a Bradley-Terry reward, watch it get hacked

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
| `16_monte_carlo.ipynb` | Tutorial 3 | Monte Carlo, importance sampling, ESS | ⭐⭐⭐ Advanced |
| `17_particle_filtering.ipynb` | Tutorial 3 | Particle filters, sequential inference | ⭐⭐⭐ Advanced |
| `18_markov_chain_monte_carlo.ipynb` | Tutorial 3 | Metropolis–Hastings, Gibbs, mixing | ⭐⭐⭐ Advanced |
| `19_sampling_the_mind.ipynb` | Tutorial 3 | MCMC with People, Kemp hierarchy sampler | ⭐⭐⭐⭐ Expert |
| `20_statistical_decision_theory.ipynb` | Tutorial 3 | Loss, risk, Bayes vs minimax, one-and-done | ⭐⭐⭐ Advanced |
| `21_markov_decision_processes.ipynb` | Tutorial 3 | MDPs, Bellman, value iteration, simulation | ⭐⭐⭐ Advanced |
| `22_q_learning.ipynb` | Tutorial 3 | Q-learning, reward shaping, Dyna, MCTS | ⭐⭐⭐⭐ Expert |
| `23_inverse_rl_goal_inference.ipynb` | Tutorial 3 | Inverse RL, goal inference, softmax, ToM = IRL | ⭐⭐⭐⭐ Expert |
| `24_pomdps_belief_inference.ipynb` | Tutorial 3 | POMDPs, belief, α-vectors, BToM, legibility | ⭐⭐⭐⭐ Expert |
| `25_modern_rl_world_models.ipynb` | Tutorial 3 | RLHF/DPO as IRL, world models, LLM ToM | ⭐⭐⭐⭐ Expert |
| `vectors_and_spaces.ipynb` | Part VIII | Vectors, dot product, cosine, matrices, ReLU | ⭐⭐ Beginner |
| `capstone_from_rules_to_weights.ipynb` | Part VIII | Capstone: ridge vs learned features vs kernel ridge | ⭐⭐⭐ Intermediate |
| `capstone_neural_net_fundamentals.ipynb` | Part VIII | Capstone: XOR → real digit classification | ⭐⭐⭐ Intermediate |
| `capstone_transformers_attention.ipynb` | Part VIII | Capstone: attention from scratch, induction heads | ⭐⭐⭐⭐ Expert |
| `capstone_llms_in_context_learning.ipynb` | Part VIII | Capstone: run the ICL-is-Bayesian debate | ⭐⭐⭐⭐ Expert |
| `capstone_world_models_imagination.ipynb` | Part VIII | Capstone: learn a world model, plan by imagination | ⭐⭐⭐⭐ Expert |
| `capstone_adversarial_examples.ipynb` | Part IX | Capstone: FGSM, robustness curve | ⭐⭐⭐ Intermediate |
| `capstone_fairness_formalisms.ipynb` | Part IX | Capstone: fairness criteria + impossibility | ⭐⭐⭐⭐ Expert |
| `capstone_bias_in_data.ipynb` | Part IX | Capstone: WEAT + debiasing leaks | ⭐⭐⭐ Intermediate |
| `capstone_alignment_safety.ipynb` | Part IX | Capstone: reward modeling + reward hacking | ⭐⭐⭐⭐ Expert |

---

**Need help?** Return to the [main tutorial page](../) or consult the [glossary](../glossary/) for term definitions.

**Enjoying the notebooks?** This tutorial series is generously funded by the [Japanese Probabilistic Computing Consortium Association ([JPCCA](https://jpcca.org/))](https://jpcca.org/).
