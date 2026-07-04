# CAPSTONES.md — retrospective capstone map for Parts I–VII

Per the **Capstone convention** in `ROADMAP.md` (Joe, 2026-07-04): every
chapter ends with a **Capstone** box — a mini-project with its own Colab
notebook (`notebooks/capstone_<slug>.ipynb`), modeled on the class
assignments (clusters / generalization / mc / rl — scaffolded setup + data +
TODO cells, solutions kept separate like the class `solutions/` pattern),
and/or a *real paper* replicated in miniature. The convention applies to
Phase D onward and is **retrofit to earlier Parts during their content
passes** (Phases A–C); this file is the retrofit map for the existing
chapters in Parts I–VII. **Part VIII capstones are already defined in
`ROADMAP.md`** (the per-chapter paper-anchor list in the Capstone-convention
section); Part IX's will be scoped in Phase E.

Ground rules used here: anchors are honest (no paper is credited with a
result it doesn't contain); paper anchors prefer the course reading list
(`course/readings_map.yml`); every chapter gets a capstone even if it is
assignment-flavored rather than paper-flavored; and the four class
assignments map to their natural chapters — **clusters →
`learning/mixture-models`**, **generalization → `structure/generalization`**,
**mc → `sampling/monte-carlo` + `sampling/mcmc`** (the assignment splits
naturally across the two), **rl → `decisions/q-learning`** — each marked
"adapt the class assignment." Difficulty uses `notebook_guide.md`'s scale:
⭐ Beginner · ⭐⭐ Intermediate · ⭐⭐⭐ Advanced · ⭐⭐⭐⭐ Expert. Premises follow
`CHIBANY_CANON.md` (they/them; Chibany measures, never explains; the Bento
Journal is the data artifact; Jamal codes, Alyssa explains, Tanaka-san is
wrong in interesting ways).

*Note:* `genjax/local-install.md` is an optional setup appendix, not a
chapter in the Part II arc (it's absent from the Part's chapter flow), so it
gets no capstone.

---

## Part I — Foundations (5 chapters)

### foundations/hungry
- **Capstone:** *Start your own Bento Journal.* Before any formula: pick something in your own day that repeats and is uncertain (what's for lunch, whether the bus is late, which vending machine sells out) and log it for a week, the way Chibany logs bentos. Then pose three questions about your data strictly in the chapter's three-question form — *What's possible? What am I interested in? Count them!* — and answer one by hand-counting. Scaffolded notebook provides the journal template and one worked Chibany example (tonkatsu days out of possible days).
- **Anchor:** assignment-flavored (the book's opening habit: probability questions as counting questions; no paper).
- **Notebook:** `capstone_hungry.ipynb`
- **Difficulty:** ⭐
- **Status:** todo

### foundations/probability-as-counting
- **Capstone:** *A week of two-meal days.* Build the full outcome space for a week of Chibany's lunch/dinner pairs, compute event probabilities by counting subsets ("at least one tonkatsu day," "never two hamburgers in a row"), then verify every hand-counted answer with a simple simulation loop. Closes with the birthday-problem structure in bento clothing: how many students before two bring identical bento pairs?
- **Anchor:** assignment-flavored (warm-up for the counting arithmetic the clusters assignment later leans on).
- **Notebook:** `capstone_probability-as-counting.ipynb`
- **Difficulty:** ⭐
- **Status:** todo

### foundations/conditional-probability
- **Capstone:** *The crossing-out machine.* Implement conditioning literally as the chapter teaches it: start from an enumerated outcome grid and *delete* the outcomes the evidence rules out, recomputing ratios from what survives. Use it to settle three famously slippery puzzles — the chapter's own second-meal question (with the lunch-first convention box, and Tanaka-san's $\{TT\}$ reading as the contrast case), the two-child problem, and Monty Hall — each solved purely by crossing out and counting, then checked by simulation.
- **Anchor:** assignment-flavored (set-based conditioning; the Tanaka-san contrast exercise from the canon retcon list).
- **Notebook:** `capstone_conditional-probability.ipynb`
- **Difficulty:** ⭐
- **Status:** todo

### foundations/bayes-rule
- **Capstone:** *The taxicab, three ways.* Replicate the chapter's Taxicab Problem as an experiment: (1) the exact Bayes answer by arithmetic, (2) a "population of nights" simulation — generate thousands of cab sightings and count witness-said-blue nights — and (3) the intuitive answer most people give (~80%). Sweep the base rate and witness accuracy to map exactly when intuition and Bayes diverge most, and write a short cell explaining the divergence to Tanaka-san.
- **Anchor:** the base-rate-neglect literature the chapter already cites — Kahneman & Tversky (1972) and Bar-Hillel (1980), who showed people's judgments on cab-style problems underweight the base rate relative to the Bayesian answer. (The capstone reproduces the *normative* analysis and the size of the gap, not the human experiments.)
- **Notebook:** `capstone_bayes-rule.ipynb`
- **Difficulty:** ⭐⭐
- **Status:** todo

### foundations/glossary
- **Capstone:** *Chibany's field checklist.* A cumulative self-test rather than a new topic: the notebook deals a fresh, randomly generated mini-dataset (a month of bento outcomes) and walks term-by-term through the Part I glossary — sample space, event, conditioning, prior, likelihood, posterior — requiring one small computation on the fresh data per term. Every answer is auto-checkable, so it doubles as a "am I ready for Part II?" gate.
- **Anchor:** assignment-flavored (self-assessment; no paper).
- **Notebook:** `capstone_glossary.ipynb`
- **Difficulty:** ⭐
- **Status:** todo

---

## Part II — The Tools: GenJAX (8 chapters)

### genjax/getting-started
- **Capstone:** *Hello, simulation.* The smallest complete loop: open Colab, run the standard setup cell, simulate 1,000 of Chibany's lunches at the canonical 70/30, plot the first histogram, and save a copy to Drive. The one TODO with teeth: change the probability, predict the histogram *before* re-running, then check.
- **Anchor:** assignment-flavored (environment setup + the predict-then-run habit).
- **Notebook:** `capstone_getting-started.ipynb`
- **Difficulty:** ⭐
- **Status:** todo

### genjax/why-genjax
- **Capstone:** *One model, three tools.* Take a single small model (Beta-Binomial over a student's tonkatsu rate) and express it three ways: hand-rolled numpy, GenJAX, and a sketched Stan/PyMC version (provided, read-only). Then hit the chapter's honest dividing line: modify the model so the *structure itself* is random (a CRP-style "unknown number of bento types" teaser) and observe which of the three tools can still express it directly — and which would need hand-flattening.
- **Anchor:** assignment-flavored (the chapter's own thesis made runnable: unbounded structure and models-as-values are where a PPL earns its keep, and fixed-structure continuous models are where Stan/PyMC shine).
- **Notebook:** `capstone_why-genjax.ipynb`
- **Difficulty:** ⭐⭐
- **Status:** todo

### genjax/python-basics
- **Capstone:** *Wrangle the Journal.* Load a provided semester of Bento Journal data (dates, weights, bringer, contents-when-labeled) as arrays; compute counts, means, and per-student tallies; and make two plots. Pure Python/JAX-numpy drill — every operation is one the later chapters use, so the notebook is a personal reference sheet by the end.
- **Anchor:** assignment-flavored (the exact array idioms the four class assignments' stencils assume).
- **Notebook:** `capstone_python-basics.ipynb`
- **Difficulty:** ⭐
- **Status:** todo

### genjax/first-model
- **Capstone:** *Model your own morning.* Design a generative model of a routine you know well — three chained random choices (e.g., alarm heard? → breakfast type → on-time?), in the pattern of Chibany's weather-then-bento day. Compute the outcome probabilities by hand from the chain, then simulate 10,000 mornings and verify the frequencies match. The deliverable is the *match*, hand-math vs. simulation.
- **Anchor:** assignment-flavored (generative model = imagined days; the Part II core move).
- **Notebook:** `capstone_first-model.ipynb`
- **Difficulty:** ⭐
- **Status:** todo

### genjax/traces
- **Capstone:** *Trace forensics.* The notebook hands you bundles of traces from three *mystery* models (source hidden) and asks you to work backwards: from the choice maps alone, reconstruct each model's structure — how many random choices, what each is named, which depend on which — and then write your own model that produces statistically indistinguishable traces. Jamal's rule: a trace is a complete record of one imagined world; enough records reveal the world-maker.
- **Anchor:** assignment-flavored (reading traces is the debugging skill every later chapter assumes).
- **Notebook:** `capstone_traces.ipynb`
- **Difficulty:** ⭐⭐
- **Status:** todo

### genjax/conditioning
- **Capstone:** *The two-witness night.* Extend the chapter's taxicab model to the case it ends on: two witnesses who *disagree*. Condition on both reports, compare the posterior against single-witness nights, and run a sensitivity sweep over base rate and witness accuracy to find the settings where the second witness flips the verdict. Every conditioned answer is cross-checked against exact enumeration.
- **Anchor:** assignment-flavored, on the chapter's Kahneman–Tversky taxicab setup (the paper anchor for the taxicab itself lives with `foundations/bayes-rule`).
- **Notebook:** `capstone_conditioning.ipynb`
- **Difficulty:** ⭐⭐
- **Status:** todo

### genjax/inference-in-action
- **Capstone:** *A semester of witnesses.* The chapter (once authored — it is the Phase B stub) solves one taxicab night end-to-end; the capstone scales it: a stream of sighting reports arrives over weeks, and the posterior after each night becomes the prior for the next. Implement the sequential update with importance sampling, plot the belief trajectory, and find how many consistent reports it takes to overwhelm a skeptical prior — and how few *contradictory* ones undo it.
- **Anchor:** assignment-flavored (sequential Bayesian updating; the bridge to Part III's prior-becomes-posterior refrain).
- **Notebook:** `capstone_inference-in-action.ipynb`
- **Difficulty:** ⭐⭐
- **Status:** todo

### genjax/building-models
- **Capstone:** *Build the campus spam filter for real.* Take the chapter's toy spam filter and grow it: choose your own set of word/feature flags, fit their per-class rates from a small provided message corpus, and evaluate the conditioned model's accuracy on held-out messages. Then stress-test it — find a message your filter gets confidently wrong and diagnose *which* independence assumption failed. (Tanaka-san insists more features always help; the held-out numbers get a vote.)
- **Anchor:** assignment-flavored (naive-Bayes-style model construction; the Bayes-rule + Law of Total Probability machinery the clusters assignment formalizes).
- **Notebook:** `capstone_building-models.ipynb`
- **Difficulty:** ⭐⭐
- **Status:** todo

---

## Part III — Continuous Probability & Bayesian Learning (5 chapters)

### learning/mystery-bentos
- **Capstone:** *Means that match nothing.* Recreate the chapter's paradox from scratch and then hunt it in the wild: (1) simulate the four expectation-455g scenarios from the closing exercise and show they are indistinguishable by mean yet wildly different in histogram; (2) compute variance, range, and modality for each and build a small "which scenario am I?" classifier from those summaries; (3) find or collect one real dataset (your own journal counts) whose mean is a value the data never takes, and write the one-paragraph diagnosis.
- **Anchor:** assignment-flavored (the moment-vs-distribution lesson that motivates the whole Part; feeder intuition for the clusters assignment).
- **Notebook:** `capstone_mystery-bentos.ipynb`
- **Difficulty:** ⭐⭐
- **Status:** todo

### learning/continuous
- **Capstone:** *Everyday predictions.* Replicate Griffiths & Tenenbaum's everyday-prediction study in miniature: implement the three prior *shapes* they used (power-law, Gaussian, Erlang) over quantities like movie grosses, lifespans, and cake-baking times; compute the Bayesian posterior-median prediction "given it's lasted $t$ so far, how long in total?"; and reproduce the three qualitatively different prediction rules (multiply, fill-the-deficit, add-a-constant) that fall out of the three shapes. Compare against the human median judgments reported in the paper's figures.
- **Anchor:** **Griffiths & Tenenbaum (2006), "Optimal predictions in everyday cognition," Psychological Science** (course Week 7 presentation candidate) — showed people's predictions of durations and quantities track Bayesian posterior medians under real-world prior distributions, with the prediction *function's shape* determined by the prior's shape. (The Bayesian step is previewed here; the full continuous-updating machinery arrives two chapters later.)
- **Notebook:** `capstone_continuous.ipynb`
- **Difficulty:** ⭐⭐⭐
- **Status:** todo

### learning/gaussian
- **Capstone:** *The bell-curve audit.* Fit Gaussians to the labeled Bento Journal weights (tonkatsu ≈ 500 g, hamburger ≈ 350 g), check the 68–95–99.7 rule empirically per class, and flag journal entries more than 3σ from their label's mean (mislabels? mega-bentos?). Close with a homemade central-limit demonstration: sums of decidedly non-Gaussian draws (uniform, two-point) marching toward the bell as the number of summands grows.
- **Anchor:** assignment-flavored (Gaussian fitting + z-score reasoning; the density the clusters assignment mixes).
- **Notebook:** `capstone_gaussian.ipynb`
- **Difficulty:** ⭐⭐
- **Status:** todo

### learning/bayesian-learning
- **Capstone:** *The pointing experiment.* Replicate Körding & Wolpert's sensorimotor result in miniature: a lateral shift is drawn each trial from a Gaussian prior; midway feedback is a noisy (variably blurred) observation of it; the optimal estimate is the reliability-weighted conjugate combination of prior and likelihood. Implement the Gaussian–Gaussian update, simulate an ideal observer across feedback-blur conditions, and reproduce the paper's signature pattern: the blurrier the feedback, the harder estimates are pulled toward the prior mean. This is also the same conjugate update as clusters assignment Problem 1 — the capstone and the assignment check each other.
- **Anchor:** **Körding & Wolpert (2004), "Bayesian integration in sensorimotor learning," Nature** (course Week 8 presentation candidate) — showed human pointing under uncertainty combines a learned prior with noisy feedback, weighted by their reliabilities, as Bayes prescribes. Rhymes with **clusters assignment Problem 1** (Gaussian-Gaussian conjugate update).
- **Notebook:** `capstone_bayesian-learning.ipynb`
- **Difficulty:** ⭐⭐⭐
- **Status:** todo

### learning/mixture-models
- **Capstone:** **Adapt the class assignment (clusters).** Categorization and prediction in a 2-component Gaussian mixture on the bento weights: Problem 1's conjugate update, Problem 2's posterior-responsibility and marginal-predictive derivations via Bayes' rule and the Law of Total Probability, plus the GenJAX stencil's bonus part. The capstone version swaps in a fresh dataset (a new semester's weights, different mixing proportion) and adds one extension the class version lacks: watch the posterior over the mixing weight converge as the journal grows.
- **Anchor:** the class **clusters** assignment (`course/assignments/clusters/`), adapted.
- **Notebook:** `capstone_mixture-models.ipynb`
- **Difficulty:** ⭐⭐⭐
- **Status:** todo

---

## Part IV — Structure: Concepts, Causes & Hierarchies (6 chapters)

### structure/generalization
- **Capstone:** **Adapt the class assignment (generalization).** Build your own Bayesian generalization model for the six animals (Cow, Dolphin, Chicken, Seal, Penguin, Bat): design a hypothesis space, set a prior, compute posteriors and predictives under weak *and* strong sampling, then expand to all $2^6-1 = 63$ hypotheses to watch the No Free Lunch theorem bite. The capstone framing keeps the assignment's central lesson — there is no single correct $\mathcal{H}$; the project is about how the framework behaves under *your* choice — and adds the bundle's own golden-sticker/number-game data as a second domain.
- **Anchor:** the class **generalization** assignment (`course/assignments/generalization/`), adapted. Paper backbone: **Tenenbaum (1999, NeurIPS)** — Bayesian concept learning with the size principle (course Week 4 required reading) — and **Tenenbaum & Griffiths (2001, BBS)**, which showed **Shepard's (1987)** exponential law of generalization emerging from Bayesian inference over hypothesis sets, exactly the derivation the bundle's third part walks through.
- **Notebook:** `capstone_generalization.ipynb`
- **Difficulty:** ⭐⭐⭐
- **Status:** todo

### structure/bayes-nets
- **Capstone:** *The health-center network.* Build a five-node Bayes net for a campus scenario (season → flu; flu → fever, fatigue; late-night modeling contest → fatigue), specify its CPTs, and implement queries by enumeration: diagnostic ($P(\text{flu}\mid\text{fever})$), predictive, and intercausal. Then rebuild the same joint as one giant table and verify the factorization saves exactly the parameters the chapter promised. Includes a homage query on Pearl's classic burglary–earthquake–alarm network.
- **Anchor:** assignment-flavored; the burglary–earthquake–alarm example follows **Pearl (1988)**, the book that introduced Bayes nets (used here as the canonical worked example, not a replicated experiment).
- **Notebook:** `capstone_bayes-nets.ipynb`
- **Difficulty:** ⭐⭐⭐
- **Status:** todo

### structure/conditional-independence
- **Capstone:** *Manufacture a spurious correlation.* Explaining away, done to yourself: simulate a world where bento tastiness and bento heaviness are truly independent, but Chibany only *journals* a bento when it is tasty OR heavy — and show the selection induces a negative correlation in the journal (a collider you conditioned on by keeping records). Then write a d-separation checker for small DAGs and use it to predict, before simulating, which of six conditioning patterns create or destroy dependence.
- **Anchor:** assignment-flavored (Berkson-style selection bias + d-separation practice; the collider logic is Pearl's, the bento selection story is ours).
- **Notebook:** `capstone_conditional-independence.ipynb`
- **Difficulty:** ⭐⭐⭐
- **Status:** todo

### structure/causal-bayes-nets
- **Capstone:** *Observing vs. intervening.* Replicate the core contrast of Lagnado & Sloman in silico: generate data from a hidden three-variable causal structure and build two learners — one that only observes, and one that can *set* variables (the do-operator) and watch what changes. Score how often each recovers the true graph as data grows, and reproduce the headline: intervention breaks the symmetries observation cannot. Finish by running one `do()` on your Part-IV bento network and comparing $P(y\mid x)$ with $P(y\mid do(x))$ on a confounded pair.
- **Anchor:** **Lagnado & Sloman (2002, Proc. CogSci), "Learning causal structure"** (course Week 5 presentation candidate) — showed people learn causal structure better from their own interventions than from matched observation. (The capstone reproduces the *informational* advantage of intervention with ideal learners; the human data are the paper's.)
- **Notebook:** `capstone_causal-bayes-nets.ipynb`
- **Difficulty:** ⭐⭐⭐
- **Status:** todo

### structure/information-theory
- **Capstone:** *Twenty questions with the Journal.* Compute the entropy of Chibany's bento days from journal frequencies, then build a question-asking agent that always asks the question with maximum expected information gain ("is it over 425 g?", "did student A bring it?") and show it identifies the day's bento in ~$H$ bits' worth of questions — while a fixed question order does measurably worse. Extension: recompute everything under the collider scenario from two chapters back and watch mutual information detect the dependence the raw counts hid.
- **Anchor:** assignment-flavored (entropy, information gain, and mutual information as *tools*, in Shannon's framework; no specific experiment replicated).
- **Notebook:** `capstone_information-theory.ipynb`
- **Difficulty:** ⭐⭐⭐
- **Status:** todo

### structure/hierarchical-bayes
- **Capstone:** *Learning to learn about bags.* Replicate the marble-bags logic of Kemp, Perfors & Tenenbaum in miniature: a hierarchical Beta-Binomial where each bag (student) has its own rate $\theta_i$, tied together by population-level parameters — the mean/concentration $(\varphi, \kappa)$ parameterization the course uses. Show the two signature effects: *shrinkage* (extreme students pulled toward the population mean, by amounts that depend on their data) and the *overhypothesis* payoff (after seeing many uniform bags, ONE marble from a new bag pins down the whole bag). This model is exactly the one the mc assignment's Problem 2 builds a sampler for — here it is solved on a grid, so the MCMC chapter can be checked against it later.
- **Anchor:** **Kemp, Perfors & Tenenbaum (2007), "Learning overhypotheses with hierarchical Bayesian models," Developmental Science** (course Week 4 presentation candidate) — the hierarchical-Bayes formalization of overhypotheses (e.g., the shape bias). Feeds directly into the **mc** class assignment (Problem 2 uses this exact model).
- **Notebook:** `capstone_hierarchical-bayes.ipynb`
- **Difficulty:** ⭐⭐⭐
- **Status:** todo

---

## Part V — Chains, Walks & Sampling (7 chapters)

### sampling/markov-chains
- **Capstone:** *How shuffled is shuffled?* The chapter's card-shuffle metaphor, made quantitative: implement top-in-at-random and riffle shuffles on a small deck (5–8 cards, so the state space is enumerable), track the full distribution over orderings step by step, and measure its total-variation distance to uniform. Reproduce the famous phenomenon Bayer & Diaconis proved for real decks: the distance stays high, then falls off a cliff — a *cutoff* — rather than decaying gently; locate your small deck's cutoff and compare shuffle types. Then bring it home: verify Chibany's 70/30 tonkatsu chain reaches ITS stationary distribution the same way, by power iteration.
- **Anchor:** **Bayer & Diaconis (1992), "Trailing the dovetail shuffle to its lair," Annals of Applied Probability** — proved riffle-shuffle mixing exhibits a sharp cutoff (≈7 shuffles for 52 cards). (The capstone reproduces the cutoff *phenomenon* on a computationally tractable deck; the 52-card theorem is theirs.)
- **Notebook:** `capstone_markov-chains.ipynb`
- **Difficulty:** ⭐⭐⭐
- **Status:** todo

### sampling/random-walks-networks
- **Capstone:** *PageRank the mind.* Replicate Griffiths, Steyvers & Firl in miniature: take a small word-association network (provided subset of association norms), run PageRank over it (with the teleport/damping fix the chapter derives from the ε-trick), and test whether high-$\pi$ words predict which word people produce first when cued with a letter — against the word-frequency baseline the paper beat. Verify along the way that on the undirected version, $\pi$ ∝ degree, and that power iteration and the degree formula agree.
- **Anchor:** **Griffiths, Steyvers & Firl (2007), "Google and the mind: Predicting fluency with PageRank," Psychological Science** (course Week 6 presentation candidate; cited in-chapter) — showed PageRank over a semantic network predicts letter-cued fluency responses better than word frequency.
- **Notebook:** `capstone_random-walks-networks.ipynb`
- **Difficulty:** ⭐⭐⭐
- **Status:** todo

### sampling/memory-search
- **Capstone:** *The mindless forager.* Replicate the core of Abbott, Austerweil & Griffiths: run a plain censored random walk (report a node only on first visit) over a semantic network of animals, and show it reproduces the behavioral signatures of human fluency data — bursty within-category runs separated by switches, with the characteristic inter-response-time pattern around category switches — *without* any explicit two-stage search strategy. Compare your walk's statistics to the strategic-forager story and articulate why the simpler account survives. Extension hook: the chapter's own open problem (sampling-based posterior inference over the network's edges) is flagged as a genuine final-project seed, not part of the capstone.
- **Anchor:** **Abbott, Austerweil & Griffiths (2012), "Human memory search as a random walk in a semantic network," NeurIPS** (course Week 6 *required* reading; the chapter's spine) — showed a censored random walk on a semantic network accounts for fluency-data patterns previously attributed to strategic patch-based foraging.
- **Notebook:** `capstone_memory-search.ipynb`
- **Difficulty:** ⭐⭐⭐
- **Status:** todo

### sampling/monte-carlo
- **Capstone:** **Adapt the class assignment (mc, Problems 1 & 3).** Estimate the tail probability $P(Y>2)$ for $Y\sim N(0,1)$ by naive Monte Carlo, watch it fail (almost every sample is wasted), then rescue it with importance sampling from a shifted proposal — and quantify the rescue with the effective sample size, working through the assignment's punchline that a weight-based ESS measures *proposal quality*, not estimator accuracy. Capstone twist: repeat with a proposal that is too narrow and diagnose the silent-failure mode where ESS looks fine and the answer is wrong.
- **Anchor:** the class **mc** assignment (`course/assignments/mc/`), Problems 1 and 3, adapted. (Problem 2, the MCMC sampler, lives with `sampling/mcmc` below.)
- **Notebook:** `capstone_monte-carlo.ipynb`
- **Difficulty:** ⭐⭐⭐
- **Status:** todo

### sampling/particle-filtering
- **Capstone:** *Track the drifting mascot.* Chibany's daily arrival time at the nap spot drifts over the semester; noisy sightings trickle in one per day. Build the full weight–resample–propagate loop to track the drift online, then run the two experiments that teach the method's soul: (1) switch resampling off and watch weight degeneracy kill the swarm (one particle hoards all the weight); (2) restore resampling triggered by an ESS threshold and show the tracker survives even a sudden regime change (Chibany discovers a new nap spot).
- **Anchor:** assignment-flavored (sequential importance resampling and its failure modes; ESS reappears from the mc assignment in its streaming role).
- **Notebook:** `capstone_particle-filtering.ipynb`
- **Difficulty:** ⭐⭐⭐
- **Status:** todo

### sampling/mcmc
- **Capstone:** **Adapt the class assignment (mc, Problem 2).** Build the interleaved sampler for the hierarchical Beta-Binomial model of Kemp, Perfors & Tenenbaum (2007), in the $(\varphi,\kappa)$ parameterization: Gibbs steps for the per-bag rates (conjugate full conditionals — always accept), Metropolis–Hastings steps for the hyperparameters (propose, ratio, accept/reject). Verify against the grid-solved posterior from the hierarchical-Bayes capstone, tune the MH proposal width via acceptance rate and trace plots, and report burn-in and mixing diagnostics honestly.
- **Anchor:** the class **mc** assignment (`course/assignments/mc/`), Problem 2, adapted; model from **Kemp, Perfors & Tenenbaum (2007)**.
- **Notebook:** `capstone_mcmc.ipynb`
- **Difficulty:** ⭐⭐⭐
- **Status:** todo

### sampling/sampling-the-mind
- **Capstone:** *Be the acceptance step.* Run MCMC with People on yourself: a one-dimensional stimulus space (the chapter's stick-figure-animal spirit, or simply "the ideal bento weight"), where on each trial the notebook proposes a candidate and *you* choose between current state and proposal — your choice is the accept/reject step. Collect a few hundred trials, discard burn-in, and read out the histogram: a portrait of your own subjective distribution, recovered as a stationary distribution. Compare chains started from opposite ends of the space as a convergence check, and (optional, with a friend) compare two people's recovered distributions.
- **Anchor:** **Sanborn & Griffiths (2007, NIPS), "Markov chain Monte Carlo with people"** (course Week 7 *required* reading; the chapter's spine) — showed that a person making choices between states can serve as an MCMC acceptance rule, so the chain's stationary distribution reveals their mental category structure.
- **Notebook:** `capstone_sampling-the-mind.ipynb`
- **Difficulty:** ⭐⭐⭐⭐
- **Status:** todo

---

## Part VI — Decisions & Reinforcement Learning (6 chapters)

### decisions/decision-theory
- **Capstone:** *One and done, verified.* Replicate the argument of Vul et al.: an agent that decides by drawing $k$ samples from its posterior and taking the majority vote. Implement the reward-*rate* analysis (good decisions per unit time, with a per-sample time cost), sweep $k$, and reproduce the headline — across a wide range of realistic costs the optimal $k^*$ is tiny, often 1 — plus the corollary that a $k=1$ agent *probability matches*. Then connect back to loss functions: show how the optimal point decision shifts (mean/median/mode) as the loss changes (squared, absolute, 0–1).
- **Anchor:** **Vul, Goodman, Griffiths & Tenenbaum, "One and done? Optimal decisions from very few samples" (CogSci 2009; Cognitive Science 2014)** (course Week 7 optional reading; already developed in-chapter) — showed that under time costs, deciding from very few posterior samples is globally optimal, rationalizing probability matching.
- **Notebook:** `capstone_decision-theory.ipynb`
- **Difficulty:** ⭐⭐⭐
- **Status:** todo

### decisions/mdps
- **Capstone:** *Plan the garden before walking it.* Implement value iteration from scratch on GardenPath and watch the value function radiate backward from the goal; extract the greedy policy and verify it takes the safe L-shaped path. Then design your own gridworld (choose walls, hazards, goal, discount) and answer three planning questions exactly: where does the optimal path bend as $\gamma$ falls? which hazard penalty makes the shortcut worth it? and does the Bellman backup converge from *any* initialization? This is the model-*based* half of the rl assignment's bonus, done first and done exactly, so Q-learning's model-free route has something to be checked against.
- **Anchor:** assignment-flavored; rhymes with the class **rl** assignment's bonus (value-iteration planner on the same GardenPath world, from Ho et al. 2015).
- **Notebook:** `capstone_mdps.ipynb`
- **Difficulty:** ⭐⭐⭐
- **Status:** todo

### decisions/q-learning
- **Capstone:** **Adapt the class assignment (rl).** Q-learning on GardenPath: Problem 1's one-line TD update under outcome feedback; Problem 2's reward-hacking diagnosis — under human-style action feedback the agent finds the +14-per-lap positive cycle and loops forever instead of reaching the goal; Problem 3's fix — design a potential function $\Phi(s)$ and add potential-based shaping, verifying both that it reaches the goal and that shaping provably leaves the optimal policy unchanged (Ng, Harada & Russell, 1999); bonus — confirm the model-free route matches the value-iteration plan from the previous capstone. Optional extension for the cognitively curious: run the chapter's two-step task simulation and reproduce the model-free vs. model-based stay-probability signature of Daw et al. (2011).
- **Anchor:** the class **rl** assignment (`course/assignments/rl/`), adapted; world and feedback schemes from **Ho, Littman, Cushman & Austerweil (2015)** (course Week 8 optional reading), shaping invariance from **Ng, Harada & Russell (1999)**; optional extension replicates the stay-probability analysis of **Daw, Gershman, Seymour, Dayan & Dolan (2011)** (the two-step task, covered in-chapter).
- **Notebook:** `capstone_q-learning.ipynb`
- **Difficulty:** ⭐⭐⭐
- **Status:** todo

### decisions/inverse-rl
- **Capstone:** *Freeze-frame mind reading.* Replicate the classic inverse-planning experiment in miniature: an agent walks a gridworld toward one of several possible goals; you observe the path only up to frame $t$. Build the observer — value iteration per candidate goal, a softmax (Boltzmann) policy as the likelihood, exact posterior by enumeration — and reproduce the signature online dynamics: the posterior is flat while the path is ambiguous, then snaps to the true goal at the disambiguating turn. Sweep the rationality $\beta$ to show the same path can be uninformative ($\beta\to 0$) or damning ($\beta$ large), and exhibit the ill-posedness honestly: two different reward functions your data cannot separate.
- **Anchor:** **Baker, Tenenbaum & Saxe (2007, Proc. CogSci), "Goal inference as inverse planning"** (course Week 9 *required* reading) and **Baker, Saxe & Tenenbaum (2009, Cognition)** — the inverse-planning framework in which observers invert a probabilistic planner to infer goals from partial paths, matching human online goal judgments. (Reviewed as "ToM = IRL" by Jara-Ettinger 2019; per the chapter, the framework itself is Baker & Tenenbaum's.)
- **Notebook:** `capstone_inverse-rl.ipynb`
- **Difficulty:** ⭐⭐⭐⭐
- **Status:** todo

### decisions/pomdps-belief
- **Capstone:** *Solve the Tiger, then be legible.* Two acts. Act 1: solve the Tiger problem — implement the Bayesian belief update over growls, then find the optimal belief-threshold policy (listen while uncertain, open when confident enough) by comparing the value of listening vs. acting at each belief, and map how the thresholds move as listening accuracy and the tiger penalty change. Act 2: run the inversion one level up — reuse the inverse-planning observer on two equal-length paths to the same goal and score each frame's observer posterior, reproducing the "showing vs. doing" gap: the legible path pays a small efficiency cost to make the observer's belief converge early.
- **Anchor:** the **Tiger problem of Kaelbling, Littman & Cassandra (1998)** (the chapter's core worked example), and the showing-vs-doing legibility effect of **Ho, Littman, MacGlashan, Cushman & Austerweil (2016, NeurIPS)** (course Week 9 presentation candidate) — demonstrators systematically deviate from efficient action to make their goal easy to infer; formalized as belief-directed planning in **Ho et al. (2021, JEP: General)**.
- **Notebook:** `capstone_pomdps-belief.ipynb`
- **Difficulty:** ⭐⭐⭐⭐
- **Status:** todo

### decisions/modern-rl-world-models
- **Capstone:** *Alignment from choices.* Recover a reward function from pairwise preferences, the RLHF way: generate (or collect from a friend) a few dozen pairwise choices among snacks — Chibany choosing between tonkatsu, hamburger, and ramen — fit the Bradley–Terry reward model by Bayesian inference (prior over latent rewards, each choice a coin flip with probability $\sigma(r_i-r_j)$, posterior by importance sampling), and read off the posterior-mean rewards. Then probe the two honest failure modes: how many comparisons until the ranking stabilizes, and what happens when the chooser is inconsistent (temperature high) — the reward model confidently averages preferences the chooser doesn't stably have.
- **Anchor:** the reward-modeling core of **Christiano et al. (2017), "Deep reinforcement learning from human preferences," NeurIPS** — learning a reward model from pairwise human preferences via a Bradley–Terry choice model, then optimizing against it — replicated in miniature (the reward-model step, not the deep-RL step), exactly as the chapter frames RLHF/DPO as inverse RL.
- **Notebook:** `capstone_modern-rl-world-models.ipynb`
- **Difficulty:** ⭐⭐⭐⭐
- **Status:** todo

---

## Part VII — How Complex Should a Model Be? (3 chapters)

### complexity/bias-variance
- **Capstone:** *The Model Fair, scored honestly.* Judge the fair yourself: fit polynomial regressions of increasing degree to a noisy sample, and estimate the bias–variance decomposition *empirically* by refitting across many resampled training sets — reproducing the canonical U-shaped test error while training error only falls. Then push past the wall: a random-features regression on a genuinely high-dimensional input (this matters — low-dimensional versions don't show the effect), sweeping feature count through the interpolation threshold $p=n$ to reproduce the double-descent curve, with the minimum-norm solution's implicit regularization doing the second descent.
- **Anchor:** **Belkin, Hsu, Ma & Mandal (2019), "Reconciling modern machine-learning practice and the classical bias–variance trade-off," PNAS** — demonstrated the double-descent test-error curve unifying the classical U-shape with modern overparameterized interpolation. (The classical decomposition half is standard textbook material, not attributed to a single paper.)
- **Notebook:** `capstone_bias-variance.ipynb`
- **Difficulty:** ⭐⭐⭐⭐
- **Status:** todo

### complexity/dpmm
- **Capstone:** *How many kinds of bento are there, really?* Fit a Dirichlet process mixture to the semester's bento weights with a CRP Gibbs sampler and read off the posterior over the *number* of clusters — the question Part III's fixed-$K$ mixture could not ask. Then replicate the cognitive claim in miniature: implement Anderson's sequential local-MAP assignment rule and show (a) it is the CRP predictive rule applied greedily, (b) it exhibits order effects the full sampler doesn't, and (c) sweeping the concentration $\alpha$ slides the model between prototype-like (few clusters) and exemplar-like (every item its own cluster) categorization.
- **Anchor:** **Anderson (1991), "The adaptive nature of human categorization," Psychological Review** — the rational model of categorization, whose sequential cluster-assignment scheme was later shown (Neal 1998; Sanborn, Griffiths & Navarro 2010) to be approximate inference in exactly this DPMM. Course companion: **Gershman & Blei (2012)** tutorial (Week 10 presentation candidate).
- **Notebook:** `capstone_dpmm.ipynb`
- **Difficulty:** ⭐⭐⭐⭐
- **Status:** todo

### complexity/gaussian-processes
- **Capstone:** *The smooth-curve stall.* The Model Fair's final event: predict the bento-weight trend over the semester's remaining weeks. Fit GP regressions with different kernels (RBF at several length-scales, periodic for the weekly rhythm, linear) to the journal time series, and score each by *log predictive likelihood* on held-out weeks — so entries are rewarded for calibrated uncertainty, not just close means. Show the two behaviors that make GPs the honest entry: error bars that grow away from the data, and a marginal likelihood that automatically prefers the simpler kernel when both fit. (The infinite-width neural-network connection is deliberately left as the on-ramp — it is Part VIII's `from-rules-to-weights` capstone in ROADMAP.md.)
- **Anchor:** assignment-flavored (kernel choice, predictive log-likelihood, and Bayesian Occam's razor over kernels, in the Rasmussen & Williams GP-regression framework; no single experiment replicated).
- **Notebook:** `capstone_gaussian-processes.ipynb`
- **Difficulty:** ⭐⭐⭐⭐
- **Status:** todo
