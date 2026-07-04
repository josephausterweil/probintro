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

## Phase D — Part VIII: deep nets / LLMs  [status: pending · P1 — Week 11 lecture Jul 10]

Author the five stubbed chapters (`from-rules-to-weights`,
`neural-net-fundamentals`, `transformers-attention`,
`llms-in-context-learning` ← the ICL-as-hierarchical-Bayes payoff chapter,
`world-models-imagination`), after/with the Week 11 deck. **This is the heart
of Joe's modernization goal — re-scope against the research findings.**

## Phase E — Part IX: ethics / fairness / safety  [status: pending · P1 — Week 12 lecture Jul 17]

Author the four stubbed chapters (`adversarial-examples`,
`fairness-formalisms` as conditional-probability statements,
`bias-in-data`, `alignment-safety` with the RLHF/reward-hacking callbacks).

## Phase F — continuity + reference sync  [status: pending · P2]

Canon-ledger audit across all rewritten Parts; notebook-guide/glossary/course
lecture-card sync; retire remaining legacy phrasing.

## Phase G — Japanese regeneration  [status: pending · P2 — LAST]

Regenerate all `.ja.md` from the rewritten English (keep Shohei Yoshida's
標本空間 terminology — it is now the book-wide standard); add the 5 missing
siblings (bias-variance, gaussian-processes, inverse-rl, pomdps-belief,
modern-rl-world-models — the last of the "5 broken links"); fix the JA
glossary anchor mismatch (EN-style anchors on JA headings).

## Housekeeping  [status: pending · P3]

- Dedupe the two identical deploy workflows (`main.yml` / `hugo.yaml`) —
  they race; both now pin Hugo 0.163.3 (0.150 broke per-language aliases).
- PR #7 (Beta-Bernoulli exercise, Dec 2025) — review or close.
- `~/work/isableFastRender-quarantine` — junk build artifact; delete when
  confirmed unneeded.

---

## Decision log

- 2026-07-02: tracks = toggle+badges; FULL story-driven rewrite; JA moved
  now / regenerated last; restructure-first-then-author (Joe).
- 2026-07-04: restructure + PR #8 propagation deployed; PR #8 closed with
  attribution; course homepage follows.
- *(append here — including any re-scoping driven by the 2026-07 deep-research
  report on positioning/modernization.)*
