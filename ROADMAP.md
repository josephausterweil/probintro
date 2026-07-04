# ROADMAP — after the Part restructure (2026-07-04)

The structural refactor is **live** (9 Parts, reading tracks, aliases, canon,
shortcodes — see `CHIBANY_CANON.md`, `scripts/url_map.csv`, and the merge
commit `ebf03c8`). This file is the surviving plan for everything that came
AFTER the structure phase in the approved refactor plan, kept as an
**adjustable backlog**: each phase has a Status / Priority / Scope block so
individual pieces can be re-scoped, re-ordered, or dropped when new
information lands (e.g. the 2026-07 competitive-landscape research report).

Statuses: `pending` / `active` / `done` / `dropped`. Priorities: `P1`
(near-term, class-coupled), `P2` (core book quality), `P3` (opportunistic).
**When priorities change, edit the block — don't re-derive the plan.**

---

## Phase A — Part I content pass (Tim's fixes)  [status: pending · P2]

- Fix the 1/3-vs-2/3 "second meal" inconsistency in
  `foundations/conditional-probability.md`: lunch/dinner convention box,
  event corrected to $A=\{HT,TT\}$ (2/3), diagram highlight matched, the
  $\{TT\}$ reading kept as an explicit Tanaka-san contrast exercise.
  (JA sibling: Shohei Yoshida's terminology is already in; keep it.)
- `probability-as-counting`: concrete $P(A)=|A|/|\Omega|$ wording; reserve
  *cardinality* for counts, introduce *weight* $w(A)$.
- Merge `foundations/glossary` into the global glossary (anchor aliases).
- Widgets 1–2: `outcome-grid.html` (conditioning by crossing out outcomes),
  `bayes-area.html` (prior × likelihood → posterior as area).
- Story-driven rewrite per canon (Part I arc: *will there be tonkatsu today?*).

## Phase B — Part II (GenJAX) pass  [status: pending · P2]

- Author the two stubs for real: `local-install`, `inference-in-action`
  (taxicab end-to-end; fixes Tim's dead-link complaints for good).
- Laptop-arc reframe of getting-started / python-basics; notebook refresh
  (standalone pip cell, Part·Chapter headers, back-links).

## Phase C — Parts III–VII story-driven rewrites  [status: pending · P2]

Full story-driven rewrite (Joe's choice) per Part, each through the proven
quality loop (student-persona clarity review to ~8/10, validator green,
built-HTML check, every-time checklist). Rewrite triage lives in the plan:
A-bucket (light, 9 ch) / B-bucket (weave, 20 ch) / C-bucket (full rewrite:
gaussian, pomdps, bias-variance, gaussian-processes, + "how to read").

Per-Part interactive additions (committed 12-widget list, priority order):
`pdf-cdf-explorer` (III), `gaussian-explorer` (III), `conjugate-update` (III),
`gmm-responsibility` (III), `number-game` (IV), `explaining-away` (IV),
`do-operator` (IV), `shrinkage-explorer` (IV), `markov-stepper` (V),
`pagerank-walker` (V). Stretch: rectangle-game, entropy-dial, fluency-walk,
fairness-tradeoff.

Poll import: 28 lecture polls (Weeks 2–10) → `{{< quiz >}}` blocks; the full
poll→chapter mapping is in the approved plan (`~/.claude/plans/
tidy-purring-zephyr.md`) and the quiz shortcode already renders provenance.
`{{< lecture >}}` boxes per chapter alongside.

## Capstone convention (Joe, 2026-07-04 — applies to Phase D onward, retrofit earlier Parts during their passes)

Every chapter ends with a **Capstone** box: a mini-project with its own Colab
notebook (`notebooks/capstone_<slug>.ipynb`), modeled on the class
assignments (clusters / generalization / mc / rl — scaffolded setup + data +
TODO cells, solutions kept separate like the class `solutions/` pattern)
and/or a *real paper* replicated in miniature. Paper anchors per Part VIII
chapter: vectors → Mikolov 2013 analogy arithmetic on real (small) embeddings;
from-rules-to-weights → ridge-vs-NNGP experiment (the Week 10 bridge);
fundamentals → Minsky & Papert's XOR, then a real digit pair;
transformers → probe attention patterns in a tiny trained transformer;
**llms-in-context-learning → replicate Xie et al.'s HMM-ICL toy AND run the
Falck et al. martingale test on it — the contested lens as a runnable
experiment**; world-models → Mozer-1994-style next-note prediction.

## Why GenJAX? (Joe, 2026-07-04)  [status: active · P1]

New chapter `genjax/why-genjax.md`: the honest answer to "why not PyMC or
Stan," anchored in what this book actually does with it — (1) **the planner
in the likelihood** (Week 9 / inverse-rl chapter: value iteration + softmax
policy INSIDE the generative model, conditioned on trajectories — no
Stan/PyMC path without hand-flattening the planner), (2) **unbounded
structure** (CRP/DPMM sampled directly vs truncated stick-breaking),
(3) models as first-class JAX values (vmap/grad/jit over *models*).
Honest flip side: fixed-structure continuous hierarchical models are
BETTER served by Stan/PyMC (NUTS quality, ArviZ, diagnostics); say so.

## Phase D — Part VIII: deep nets / LLMs  [status: ACTIVE · P1 — FLAGSHIP]

Author the five stubbed chapters (`from-rules-to-weights`,
`neural-net-fundamentals`, `transformers-attention`,
`llms-in-context-learning`, `world-models-imagination`), after/with the
Week 11 deck (Jul 10).

**Re-scoped 2026-07-04 per the positioning study: this is the book's
flagship differentiator** — the "classical probabilistic modeling → LLMs"
bridge has NO from-foundations pedagogical treatment anywhere (only research
papers and one Princeton graduate-seminar lecture). Requirements:
- `llms-in-context-learning` teaches ICL-as-Bayes as a **contested lens, not
  settled fact**: present Xie et al. 2022 / Ye et al. (ICLR 2024 WS,
  "pretraining is precisely empirical Bayes") AND Falck et al.
  (arXiv:2406.00793) martingale counter-evidence, with a runnable GenJAX toy
  (small-alphabet HMM pretraining → ICL posterior) so readers can test the
  claim themselves.
- Explicit callbacks to hierarchical Bayes (Part IV) — the payoff structure
  the rest of the book earns.

## Phase E — Part IX: ethics / fairness / safety  [status: pending · P1 — Week 12 lecture Jul 17]

Author the four stubbed chapters (`adversarial-examples`,
`fairness-formalisms` as conditional-probability statements,
`bias-in-data`, `alignment-safety` with the RLHF/reward-hacking callbacks).

## Phase F — continuity + reference sync  [status: pending · P2]

Canon-ledger audit across all rewritten Parts; notebook-guide/glossary/course
lecture-card sync; retire remaining legacy phrasing.

## Phase G — Japanese  [status: pending · P2, spine polish pulled forward]

Re-scoped 2026-07-04: the JA niche above intro-stats is confirmed empty
(統計WEB stops at three Bayes lessons; main cogsci-modeling text is a 2017
translation), but machine translation won't win it.
- **G1 (pull forward, alongside Phases A–C):** human-polish the
  Beginner-track spine — ideally inviting Shohei Yoshida to continue his
  PR #8 work (his terminology 標本空間 is now the book-wide standard).
- **G2 (last, as before):** regenerate remaining `.ja.md` from the rewritten
  English; add the 5 missing siblings (bias-variance, gaussian-processes,
  inverse-rl, pomdps-belief, modern-rl-world-models — the last "5 broken
  links"); fix the JA glossary anchor mismatch (EN-style anchors on JA
  headings).
- Before G2: check the アヒル本/Stan-ecosystem flank the survey didn't cover.

## Phase H — Colleague-outreach package, then publicize  [status: pending · P1 after D/E]

Sequencing per Joe (2026-07-04): the book stays deliberately unpublicized
until polished + content-complete; then colleague feedback; broad publicity
only after that round. This phase is the outreach *package*, staged:
- **H1 — pre-outreach essentials** (land with/right after Phases D–E):
  Zenodo DOI + CITATION.cff + versioned release; a "coming from probmods?"
  mapping page (probmods is frozen since 2023-06 and Griffiths has no
  companion-site plans — the cogsci niche is durably open); a "teach with
  this book" page (course-adoption story, track structure, per-chapter
  Colab/quiz inventory); end-of-chapter exercise sets with solutions for at
  least the Beginner-track spine; PDF export if cheap.
- **H2 — colleague round** (Joe): targeted outreach for feedback + interest
  in course use; fold feedback into a revision pass.
- **H3 — public launch** (after H2): announcement pass (cogsci lists,
  relevant subreddits/HN), basic SEO (sitemap/meta/descriptions), GitHub
  Discussions as the community channel. Adoption benchmark from the study:
  BMH sits at ~28k stars in the adjacent niche; discoverability, citability,
  and exercises are the three measured gaps.

## Housekeeping  [status: pending · P3]

- ~~Dedupe the two identical deploy workflows~~ — DONE 2026-07-04
  (`main.yml` removed after a live deploy-race failure; `hugo.yaml` is the
  single pipeline, pinned to Hugo 0.163.3 — 0.150 broke per-language aliases).
- PR #7 (Beta-Bernoulli exercise, Dec 2025) — review or close.
- `~/work/isableFastRender-quarantine` — junk build artifact; delete when
  confirmed unneeded.

---

## Decision log

- 2026-07-02: tracks = toggle+badges; FULL story-driven rewrite; JA moved
  now / regenerated last; restructure-first-then-author (Joe).
- 2026-07-04: restructure + PR #8 propagation deployed; PR #8 closed with
  attribution; course homepage follows.
- 2026-07-04 (positioning study + Joe): modernization bridge confirmed as an
  unoccupied teaching slot → Phase D is the flagship, with the contested-lens
  requirement (Falck et al. counter-evidence taught alongside). probmods
  frozen since 2023-06 AND (per Joe) Griffiths has no companion-web plans →
  the cogsci-successor positioning is durable; mapping page added to H1.
  Narrative alone is not the moat (Manga-Guide precedent) → Phases A–C stay
  P2 quality work. JA niche empty above intro-stats → G1 spine polish pulled
  forward (invite Shohei Yoshida). Joe's sequencing locked in: polish +
  complete content → colleague feedback round → THEN publicize (Phase H1→H3);
  no early publicity.
- *(append here as decisions land.)*
