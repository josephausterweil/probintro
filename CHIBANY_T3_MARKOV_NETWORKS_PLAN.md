# Plan: Tutorial 3 Chapters 13–15 — Markov Chains, Random Walks on Networks, and Memory Search

**STATUS: NOT YET BUILT.** This is a planning handoff for a fresh agent to write the chapters in a new session.

**Author:** Drafted by Claude (Opus 4.8, 1M ctx) in conversation with Prof. Austerweil, 2026-06-03, at the end of the session that built the **Week 6 lecture** (Markov Chains + Networks). The lecture is done, reviewed, shipped, and live; these textbook chapters are the parallel reading track, like Ch 8–11 were for Week 5.

**Briefing for the new agent (paste this to start):**
> Build out the Markov-chains spine, Tutorial-3 Chapters 13–15, per `textbook/CHIBANY_T3_MARKOV_NETWORKS_PLAN.md`, starting with Chapter 13. Follow the same conventions as the already-built Ch 8–11 (see `textbook/CHIBANY_T3_CH7-10_PLAN.md`). Read the **Week 6 lecture deck** (`course/week06_markov_chains_networks/week6-slides.qmd`) and its **shared-outline** (`week6-shared-outline.md`) first — they are the source material. Ask Joe before moving from one chapter to the next.

---

## TL;DR for the new agent

Build **3 new chapters** (Markov-chains spine, **Ch 13–15**) in `textbook/content/intro2/`. T3 currently ends at **Ch 12 (Hierarchical Bayes)**; these append after it. There is **no existing Markov/random-walk/network content** in T3 yet.

| File | Title (draft) | Weight | Builds on | Lecture segment it parallels |
|---|---|---:|---|---|
| `13_markov_chains.md` | Markov Chains: The Future Forgets the Past | 13 | Ch 8 (Bayes nets — "Markov" factorization, directed graphs) | Segments 1–3 (Markov property, transition matrix, stationary distribution, power iteration) |
| `14_random_walks_networks.md` | Random Walks on Networks | 14 | Ch 13 | Segments 4–5 (graphs as structure, random walk on a network, π ∝ degree, PageRank) |
| `15_memory_search.md` | Memory Search as a Random Walk | 15 | Ch 14 | Segment 6 (Abbott et al. 2012 — semantic networks, the censoring function, fluency-IRT match) |

**Why 3, not more:** the lecture is one tight arc (chain → walk-on-network → memory). A natural split is Ch 13 = the chain machinery, Ch 14 = put it on a graph, Ch 15 = the cognitive payoff. **Confirm this split with Joe** — he may want the memory-search chapter folded into Ch 14, or PageRank as its own short chapter. (See "Things to ask Joe.")

Each chapter:
- **Opens with a Chibany bento scenario** (concrete, narrative — like Ch 1's mystery bentos).
- **Develops the math** at the existing T3 level — light derivations, heavy intuition + worked examples.
- **Ends with a GenJAX (or plain JAX/NumPy) section** with runnable, validated code.
- **Cross-references** neighboring chapters in the T3 style.

---

## Source material (read these first — they are already written and reviewed)

1. **`course/week06_markov_chains_networks/week6-slides.qmd`** — the 64-slide Week 6 lecture (the canonical content + worked examples + figures). THE primary source.
2. **`course/week06_markov_chains_networks/week6-shared-outline.md`** — the segment-by-segment outline (timing, key points, contingencies).
3. **`course/week06_markov_chains_networks/PLAN.md`** — the design-decision log: every choice made while building the lecture (why Chibany's chain is 70/30, why Cat is the network hub, why the censoring function matters, etc.). Read the dated change-notes at the bottom — they capture the *reasoning*, which is what the chapters should preserve.
4. **`course/week06_markov_chains_networks/make_figures.py`** — the matplotlib source for every lecture figure. The chapters can regenerate the same figures (or the agent can adapt this script). Lecture figures are theme-dark (#111111 bg); textbook figures should match the textbook's image style instead (look at existing T3 `images/`).
5. The lecture's **required reading**: `resources/readings/abbott_nips2012_randomWalk.pdf` (Abbott, Austerweil & Griffiths 2012). Section 4 (§4.1 models, §4.2 the censoring / IRT mapping) is the technical heart of Ch 15 — read it.

---

## Style rules the new agent MUST follow

These are identical to the Ch 8–11 build. **The full, authoritative version is in `textbook/CLAUDE.md` and `textbook/CHIBANY_T3_CH7-10_PLAN.md` ("Style rules" section). Read both.** The essentials:

### Frontmatter
```markdown
+++
date = "YYYY-MM-DD"   # ← today's date; update on EVERY edit (textbook CLAUDE.md is emphatic)
title = "Chapter Title"
weight = N            # 13, 14, 15
+++
```

### Voice
- **Narrate from / around Chibany.** Chiba Tech mascot, they/them, two bentos a day, loves tonkatsu, **not a professor**. Read `content/intro2/01_mystery_bentos.md` before writing.
- **Concrete scenario first, math second, code third.** Hold this order in every chapter.
- Math: `$$...$$` block, `$...$` inline. NOT `\[...\]` / `\(...\)`.
- `{{% notice style="info|warning|success|tip" title="..." %}}` boxes for sidebars.
- **≥1 mermaid diagram per chapter** (a chain, a DAG, a network). Match the dark-themed mermaid style in existing chapters.
- End every page with the JPCCA acknowledgment: `Special thanks to [JPPCA](https://jpcca.org/) for their generous support of this tutorial series.`

### Supporting characters
Ch 8–11 used **Jamal & Alyssa** (peers/labmates; Alyssa uses they/them) — NOT Ira/Yuki (those were in the *original* plan but changed during the build). **Reuse Jamal & Alyssa** for continuity unless Joe says otherwise.

### Code rules (CRITICAL — the API has moved)
- **`validate_code_blocks.py` EXECUTES every block and compares stdout to the `**Output:**` block.** Run it before committing; the pre-commit hook + GitHub Action enforce it.
- **GenJAX 0.10.3 idioms** — see the detailed list in `textbook/CLAUDE.md` (bottom) and the memory note `feedback_genjax_block_workflow.md`. Key traps: `ChoiceMap.d({...})`, arg order `(key, constraints, args)`, `binomial` needs float `n`, no `for i in range(model_arg)`.
- Per-block directives: `<!-- validate: skip -->` (illustrative), `<!-- validate: skip-output -->`, `<!-- validate: tol=0.05 -->` (**use for stochastic Monte-Carlo cells** — random-walk simulations WILL wobble between seeds), `<!-- validate: reset -->`.
- **Never paste an `**Output:**` value you didn't execute.** Run the literal cell, paste literal stdout, set `tol` to cover the seed's real wobble.
- **Claim "green" only after a fresh foreground validator run shows 0 failures on that file.**
- *Note:* much of the Markov-chain math is **plain linear algebra** (transition matrices, `v @ P`, power iteration, `nx.pagerank`), not generative-model GenJAX. That's fine — use `jax.numpy` / `numpy` / `networkx` where it's the natural tool, and reserve `@gen` for genuinely generative pieces (e.g., the Chibany bento chain as a generative sequence, the censored-walk simulator). **Raise with Joe whether to lean GenJAX or plain-JAX for these chapters** — the content is more "simulate + linear algebra" than "condition a generative model."

---

## Per-chapter plan

> These mirror the lecture's segments. The lecture already solved the hard pedagogy (what example, what order, what to cut) — **preserve its choices**. Section lists below are a starting point; refine against the qmd.

### Chapter 13: Markov Chains — The Future Forgets the Past

**Goal:** Introduce the Markov property, transition matrices, and the stationary distribution via power iteration — all on Chibany's bento chain. Textbook parallel of Week 6 segments 1–3.

**Chibany scenario (from the lecture):** Students now bring Chibany **both** a tonkatsu and a hamburger bento each day, and Chibany **chooses** which to eat. There's a habit in the choosing: Chibany loves tonkatsu, so after a tonkatsu day they usually want it *again*, but occasionally fancy a change; after a hamburger day they almost always swing back to tonkatsu. *(This is a deliberate narrative shift from earlier chapters where bentos were brought and the contents inferred — call it out, as the lecture does.)*

**Sections (draft):**
1. **A habit you can draw.** Today's bento depends only on yesterday's → the Markov property $P(X_{t+1}\mid X_t,\dots,X_0)=P(X_{t+1}\mid X_t)$. The "Really past / Past / Less past" intuition (the lecture uses Andrey Markov portraits — for the textbook, just prose + a diagram). Connect to Ch 8: a Bayes net used the word "Markov" for a factorization; here it's the same independence idea **indexed by time**.
2. **Two views: picture and matrix.** The 2-state chain as a state diagram AND a transition matrix $P=\begin{pmatrix}0.65&0.35\\0.82&0.18\end{pmatrix}$ (rows = from T / from H; row-stochastic). *Use these exact numbers — they give a 70/30 stationary, matching Chibany's canonical "loves tonkatsu, 70/30" prior.*
3. **The matrix is a sampler.** One step = draw $u\sim\text{Uniform}(0,1)$, compare to the row (e.g. $u=0.42<0.65\Rightarrow$ stay T). The matrix + a stream of random numbers generates the whole sequence — *this is the seed of Monte Carlo (Week 7 / a later chapter).*
4. **Run it: what a chain produces.** Sample sequences (mostly T, brief H interruptions); long-run ≈ 70% T.
5. **The stationary distribution.** Define $\pi$: the long-run fraction of time in each state; $\pi P = \pi$. Card-shuffling as the motivating "what's the goal of shuffling?" example (→ uniform over orderings = a distribution that stops changing = stationarity). *The lecture opens the stationary segment with card-shuffle process-first; consider doing the same.*
6. **Finding π by power iteration.** Multiply by $P$ repeatedly; two different starts converge to the same 70/30; the chain "forgets" its start (ergodicity, defined in one line). Mention $\pi$ = left eigenvector with eigenvalue 1 as a named fact (do NOT bring in PCA/SVD — the lecture deliberately cut that).
7. **A second example.** The 3-state matrix $A=\begin{pmatrix}0&0.1&0.9\\0.5&0&0.5\\0.8&0.2&0\end{pmatrix}$ (stationary ≈ (0.42, 0.13, 0.45)) to show the method generalizes. *This is the SP25 quiz matrix; the assignment uses it.*
8. **Code section.** `jax.numpy`/`numpy`: build $P$, do power iteration, show convergence; a small generative `@gen` Chibany-chain step is optional. (Mark stochastic cells `tol=...`.)

**Forward pointer:** "Next chapter: put the states on a graph." Backward: Ch 8 (graphs + the Markov factorization).

### Chapter 14: Random Walks on Networks

**Goal:** Graphs as structure; a random walk = a Markov chain whose states are nodes; $\pi_i \propto \deg(i)$ for an undirected graph; PageRank as the same idea at scale. Week 6 segments 4–5.

**Sections (draft):**
1. **What's a graph?** $G=(V,E)$, nodes + edges, directed/undirected/weighted. Callback to Ch 8 (a Bayes net is a directed graph — there an edge meant "depends on," here "is related/connected to").
2. **Graphs are everywhere.** A node/edge table across domains (semantic network, the Web, co-authorship, social, road map, brain). Land on the **semantic network** as the one we build on.
3. **From a graph to a transition matrix.** Adjacency matrix $L$ (show it explicitly — the lecture has a dedicated "graph as a matrix" slide with the labeled $L$ for the 6-node animal network: Dog/Wolf/Cat/Lion/Tiger/Zebra). Row-normalize $L \to$ transition matrix $P$ → a walker stepping to a random neighbour.
4. **Take a walk.** A worked trace on the animal network (Wolf→Dog→Cat→Lion→Tiger→Zebra; Cat is the bridge between the "pets" and "big animals" clusters). The visited sequence IS a Markov chain.
5. **The stationary distribution of a walk.** $\pi_i \propto \deg(i)$ for undirected/unweighted graphs. Cat (the bridge, degree 4) is the most-visited node — no eigen-solve needed; the degree IS the answer.
6. **PageRank.** The stationary distribution of a random walk over a (directed) link graph — exactly the $\pi$ from Ch 13, at web scale. Griffiths, Steyvers & Firl (2007), *Google and the mind*: PageRank over a semantic network predicts human word-fluency.
7. **Code section.** `networkx` for the animal graph + `nx.pagerank`; row-normalize the adjacency matrix; simulate a walk and verify visit-frequency ∝ degree (stochastic → `tol`). *Lecture figures: `animal_net_base.png`, `animal_net_degree.png`, `pagerank.png`, `er_vs_scalefree.png`.*

**Note:** The lecture also has a short "network properties" beat (degree, shortest path, diameter; Erdős–Rényi vs. scale-free). Fold the useful bits in; don't over-expand.

### Chapter 15: Memory Search as a Random Walk

**Goal:** The cognitive payoff — Abbott, Austerweil & Griffiths (2012). Human semantic fluency = a **censored** random walk on a semantic network, reproducing the optimal-foraging IRT signature with one process. Week 6 segment 6 (the required reading). **This is the most conceptually rich chapter — get the censoring right; it's the crux and the connection to data.**

**Sections (draft):**
1. **The phenomenon.** "List as many animals as you can." People recall in **bursts by category** with switches between them (Bousfield & Sedgewick; Troyer et al.).
2. **The model.** Semantic memory is a network; recall is a random walk on it. Clusters = the walk lingering in a community; switches = crossing a bridge edge. (Same structure + process + behaviour split from Marr.)
3. **The catch: the walk ≠ the list.** A random walk **revisits** nodes and wanders through **non-animals** — a fluency list has neither. *(The lecture makes this the pivot; so should the chapter.)*
4. **The censoring function (THE mechanism).** You report a word **only the first time the walk hits it, and only if it's an animal**; everything else is censored. $\tau(k)$ = first hitting time of the $k$-th unique animal. **IRT$(k) = \tau(k) - \tau(k-1) + \text{len}(\text{word})$.** Walk through the paper's example: `animal → dog → house → dog → cat` → reported "dog, cat" (house + the second dog censored); IRT(cat) = $5-2+3 = 6$. *(Lecture figure: `censoring.png`.)*
5. **The result.** The censored walk reproduces the human optimal-foraging IRT curve (first word of a new patch is slowest = the "switch cost") **with no explicit switch rule** — one process, not two. Human-vs-model bar chart (Abbott Fig 1a vs Fig 3; lecture figure `irt_patch_switch.png`).
6. **Optional / forward:** the network as a measurement instrument — invert the walk to estimate someone's semantic network (U-INVITE), and clinical applications (Zemla & Austerweil 2019: AD networks have smaller mean degree, higher edge density, less small-world). *These are "optional extension" material in the lecture; keep light in the chapter or defer.*
7. **Code section.** Simulate a random walk on a small semantic network; apply the censoring function (first-hit-of-each-unique-animal); compute IRTs; show the patch-switch curve emerge. This is the chapter's payoff cell — make it runnable + validated (stochastic → `tol`; or fix the seed and `skip-output` if the curve shape is the point).

**Forward pointer:** Monte Carlo / sampling (Week 7) — "running a chain to estimate a distribution is Monte Carlo; next we design chains to sample on purpose (MCMC)."

---

## Cross-chapter assets to create / update

### `_index.md` updates
After the chapters are written, update `content/intro2/_index.md`:
1. Bump the `date`.
2. Extend the mermaid learning-path graph (currently ends `… → K[11. Information Theory] → L[12. Hierarchical Bayes]`):
   ```mermaid
   … L[12. Hierarchical Bayes] --> M[13. Markov Chains]
   M --> N[14. Random Walks on Networks]
   N --> O[15. Memory Search]
   ```
3. Add the three new section blurbs in the existing format.
4. Prereqs: Ch 13 ← Ch 8 (graphs/Markov factorization); Ch 14 ← Ch 13; Ch 15 ← Ch 14.

### Images
The lecture's figures (`course/week06_markov_chains_networks/images/`) are dark-themed for the slides. For the textbook, either (a) regenerate via an adapted `make_figures.py` in the textbook's image style, or (b) write the chapters with mermaid diagrams + describe-in-prose and flag a TODO for Joe to commission Chibany illustrations later. **Do NOT block on illustrations** (same rule as Ch 8–11).

### Validation & tests
- Run `python validate_code_blocks.py` after EVERY chapter; resolve all failures before committing (pre-commit hook enforces).
- Optional: one `test_chNN_code.py` per chapter exercising the canonical function (pattern after existing `test_ch5_code.py`).

---

## Suggested order of operations for the new agent

1. **Read `textbook/CLAUDE.md` + `textbook/CODE_VALIDATION.md` in full.**
2. **Read `textbook/CHIBANY_T3_CH7-10_PLAN.md`** (the proven sibling plan + its "decisions taken during the build" notes) — this Markov plan deliberately mirrors it.
3. **Read the Week 6 lecture sources** (the 5 files in "Source material" above) — especially the qmd + PLAN.md change-notes.
4. **Read `content/intro2/01_mystery_bentos.md`, `05_mixture_models.md`, and a recent built chapter (`08_bayes_nets.md` or `13`'s prereq context)** for the Chibany-narrative + math + code template.
5. **Confirm the GenJAX/JAX API** with a throwaway script before writing code (the API has moved; see CLAUDE.md). Decide GenJAX-vs-plain-JAX with Joe.
6. **Write Ch 13 first**, validate, **ask Joe for review before Ch 14.** Then 13 → 14 → 15.
7. **Student-audit each chapter** (re-read as a student who's read only up to that chapter) before committing — this caught real gaps in the Ch 8–11 build.
8. **Update `_index.md`** (incrementally or once at the end).
9. **Update dates on every modified file.** Run the validator after every chapter.
10. **Commit + push** to `origin/main` (the textbook is its own repo: `github.com/josephausterweil/probintro`, worked on inside `spring2026/textbook/`). The textbook deploys separately from the course site.

---

## Things to ask Joe about before writing

1. **Chapter split.** 3 chapters (chain / walk-on-network / memory) vs. fewer (fold memory into the network chapter) vs. more (PageRank as its own short chapter). The lecture is one arc; the textbook split is a judgment call.
2. **GenJAX vs. plain JAX.** Much of this content is linear algebra + simulation (transition matrices, power iteration, `nx.pagerank`, censored-walk simulation), not "condition a generative model." Should the chapters lean GenJAX (for consistency with Ch 8–11) or use plain `jax.numpy`/`networkx` where natural? **This is the biggest open decision.**
3. **How far into the cognitive/clinical material** (Ch 15). The Abbott censoring + IRT result is core. The Zemla 2019 AD network-stats + U-INVITE inversion are "optional extension" in the lecture — include fully, lightly, or defer to a forward pointer?
4. **Notebook companions.** Should each chapter get a Colab notebook (`textbook/notebooks/` + the standard `[📓 Open in Colab: …]` link), or does the code live only in the markdown? (Ch 8–11 varied.)
5. **Numbering.** Confirm Ch 13–15 (these append after Ch 12 Hierarchical Bayes). If Joe plans other chapters to slot in between, adjust.
6. **Relationship to Week 7 (Monte Carlo).** The lecture ends by foreshadowing MCMC. Should Ch 15 forward-point to a future Monte-Carlo chapter, and is that chapter planned? (Week 7's lecture is not yet built; its textbook chapters would be a *separate* future handoff.)

---

## What this plan does NOT cover

- **No MCMC / Metropolis-Hastings / Gibbs.** That's Week 7 (Monte Carlo) — a separate future chapter set. Ch 15 only forward-points to it.
- **No hidden Markov models (HMMs) / state-space / particle filters.** Out of scope; the chains here are fully-observed.
- **No spectral graph theory beyond "π is the leading eigenvector."** Keep eigen-machinery to a named fact.
- **No network structure-learning / community-detection algorithms.** The networks are given (or estimated via the cited U-INVITE, mentioned not derived).
- **No exercises-with-solutions enumerated here.** Draft light exercises matching the chapter level, as the existing T3 chapters do.

---

## Handoff complete

A fresh agent, given **(a)** this file, **(b)** read-access to the repo (the Week 6 lecture under `course/week06_markov_chains_networks/` + the textbook under `textbook/`), and **(c)** the briefing at the top, has everything needed to begin. Start with Ch 13, ask Joe before Ch 14.

— Drafted by Claude (Opus 4.8, 1M ctx) with Prof. Austerweil, 2026-06-03.
