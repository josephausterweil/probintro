# CLAUDE.md - Probability Tutorial Project

This file provides guidance to Claude Code when working with this repository.

## Critical Reminder: Always Update Dates

**IMPORTANT**: Whenever you modify any content file, ALWAYS update the `date` field in the frontmatter to the current date.

### How to Update Dates

1. After editing any content file, check for a `date` field in the frontmatter (the `+++` section at the top)
2. Update it to today's date in format: `date = "YYYY-MM-DD"`
3. If a file doesn't have a date field but has frontmatter, add one

### Files to Check

When modifying content, always check these locations for date fields:
- `content/intro2/_index.md` - Tutorial 3 main page
- `content/genjax/_index.md` - Tutorial 2 main page
- `content/intro/_index.md` - Tutorial 1 main page
- `content/notebook_guide.md` - Notebook guide page
- `content/glossary.md` - Glossary page
- Any other `_index.md` or standalone `.md` files in content/

### Example

```markdown
+++
date = "2025-12-05"  # ← ALWAYS UPDATE THIS TO TODAY
title = "Page Title"
weight = 3
+++
```

### Command to Find Date Fields

```bash
cd /home/jausterw/work/tutorials/amplifier_play/probintro
grep -r "^date = " content/ --include="*.md"
```

### Automated Date Update

After editing files, run:
```bash
# Get today's date
TODAY=$(date +%Y-%m-%d)

# Example: Update a specific file
sed -i "s/^date = \".*\"/date = \"$TODAY\"/" content/intro2/_index.md
```

## Project Structure

This is a Hugo-based tutorial website for probability and probabilistic computing.

### Key Directories

- `content/` - Markdown content files organized by tutorial
  - `intro/` - Tutorial 1: Discrete Probability
  - `genjax/` - Tutorial 2: GenJAX Programming
  - `intro2/` - Tutorial 3: Continuous Probability & Bayesian Learning
- `notebooks/` - Jupyter notebooks (linked via symlink in static/)
- `static/` - Static assets (CSS, images, notebooks symlink)
- `public/` - Hugo build output (generated)

### Important Files

- `hugo.toml` - Hugo configuration
- `content/notebook_guide.md` - Notebook reference page (weight: 99)
- `content/glossary.md` - Glossary page (weight: 100)

## Notebook Links

All notebook links should use Google Colab URLs in this format:
```markdown
[📓 Open in Colab: `notebook_name.ipynb`](https://colab.research.google.com/github/josephausterweil/probintro/blob/main/notebooks/notebook_name.ipynb)
```

## Hugo Development

- Hugo server runs on: `http://localhost:1313/probintro/`
- Check if running: `ps aux | grep hugo`
- Public output: `public/` directory

## Math rendering (MathJax) — the `\{` / set-brace gotcha (FIXED)

Math is MathJax, delimiters `$...$` (inline) and `$$...$$` (block). There is a recurring
trap: by default Goldmark (Hugo's markdown parser) processes the text *inside* `$...$`
**before** MathJax sees it, and escapes backslash-punctuation — so `\{` collapses to a bare
`{` (an invisible MathJax grouping brace) and set notation like `$\{a, b\}$` renders with **no
visible braces** (`\\`, `\%`, `_` can misbehave too).

**This is fixed at the source:** `hugo.toml` enables the Goldmark **passthrough** extension
(`[markup.goldmark.extensions.passthrough]`) with delimiters matching MathJax, so math content
is now handed to MathJax untouched. `$\{a, b\}$` renders correctly; you do **not** need to use
`\lbrace`/`\rbrace`. Do not remove that config. The one caveat passthrough imposes: **never put
a literal prose dollar sign next to text** (write "5 dollars", not "$5") — `$` is now a math
delimiter at build time, so a stray `$...$` pair in prose would render as math. (All current
chapters use `$` only for math, which is why this was safe to enable.)

## Git Workflow

This project is hosted at: `git@github.com:josephausterweil/probintro.git`
Main branch: `main`

When committing changes, include date updates:
```bash
git add content/
git commit -m "Update content and dates for [feature/fix]"
git push origin main
```

## Special Notes

- The `notebooks/` directory is symlinked into `static/notebooks` so Hugo serves them
- Page weights determine sidebar order (lower = higher in menu)
- Tutorial sections use `_index.md` for landing pages
- Individual chapters are separate `.md` files

## Code Validation

**CRITICAL**: All Python/JAX code blocks in tutorials MUST be validated before committing.

### Automated Validation

The repository has automated validation that runs:
1. **Pre-commit hook** - Validates code blocks before allowing commits
2. **GitHub Actions** - Validates on push/PR to main branch

### Manual Validation

Run validation anytime:
```bash
python validate_code_blocks.py
```

### Key Rules for Code Blocks

1. **Python Syntax** - All code must parse correctly
2. **JAX Compatibility** - Use `jnp.where()` instead of Python `if/else`
3. **Complete Imports** - Include all required imports (jnp, jr, genjax)

### When Writing New Code Blocks

```python
# ✅ Good - Complete and correct
import jax.numpy as jnp
import jax.random as jr
from genjax import generative as genjax

key = jr.PRNGKey(0)
x = jnp.array([1.0, 2.0, 3.0])
result = jnp.where(x > 1.5, x, 0.0)

@genjax.gen
def my_model():
    return jnp.normal() @ "x"
```

```python
# ❌ Bad - Missing imports, using if/else
x = jnp.array([1, 2, 3])  # jnp not imported!
if x > 0:  # Should use jnp.where()!
    result = x
```

See `CODE_VALIDATION.md` for complete documentation.

## Funding Acknowledgment

This project is generously funded by the Japanese Probabilistic Computing Consortium Association (JPCCA).
Always include this acknowledgment in new pages.

## Working on GenJAX code blocks (read before editing any code cell)

`validate_code_blocks.py` now **executes** every Python block and **compares its stdout against the
`**Output:**` block** — not just syntax. Run it before committing; a block can be syntactically valid and still
crash at runtime or print numbers that no longer match the prose. Default scope executes `intro2/` and
`genjax/`; `--exec-all` adds `intro/` + `glossary.md`; `--no-exec` is the old syntax-only behavior; pass file
paths to scope to specific files.

**Per-block directives** (HTML comment on the line *immediately before* the ` ```python ` fence):
- `<!-- validate: skip -->` — don't execute (still syntax-checked). For illustrative fragments / pseudo-code /
  "Good vs Bad" snippets / line-by-line walkthrough pieces that can't run standalone.
- `<!-- validate: skip-output -->` — execute, but don't compare stdout.
- `<!-- validate: tol=0.05 -->` — compare numbers with that absolute tolerance. **Use for stochastic
  Monte-Carlo cells** whose printed estimate wobbles between runs but should stay near its reference.
- `<!-- validate: reset -->` — run this block in a fresh namespace (drop prior state).

Blocks in one file share a namespace and run in order (so a continuation cell may use names from an earlier
cell); a page **bundle** (a dir with `_index.md` + weighted parts, e.g. `07_generalization/`) shares one
namespace across its parts.

**GenJAX 0.10.3 idioms** (verify in a throwaway script before editing — the API has moved; the older chapters
were written against a dead API):
- `from genjax import gen, flip, normal, beta, binomial, categorical, uniform, ChoiceMap` — NOT `simulate`,
  NOT `choice_map`, and never `jnp.normal` / `jnp.bernoulli` (those are fake).
- run: `tr = model.simulate(key, args_tuple)` — `args` is required (`()` if none).
- condition: `cm = ChoiceMap.d({"addr": value})`; then `model.generate(key, cm, args)` or
  `model.importance(key, cm, args)` → both return `(trace, log_weight)`. **Arg order is `(key, constraints,
  args)`** — putting the args tuple in the constraint slot throws a beartype "tuple violates ChoiceMap" error.
- read: `tr.get_retval()`, `tr.get_choices()["addr"]`, `tr.get_score()`.
- `binomial(n, theta)` needs `n` as a **float** (int n + float theta → dtype error).
- `for i in range(n)` where `n` is a model **argument** → `TracerIntegerConversionError`. Make the count a
  Python constant captured in the `@gen`, or use a factory closure. This is a pedagogy decision — raise it.

**Rules of the road** (learned the hard way):
1. **Never paste an `**Output:**` value you didn't execute.** Run the literal cell, paste literal stdout. For
   stochastic cells, run, then set `tol` to cover the seed's real wobble. Don't hand-derive "obvious" numbers.
2. **Keep the teaching for-loop style** in older chapters until a later chapter introduces `vmap`; fix only the
   broken API, not the pedagogy, unless asked.
3. **Claim "green" only after a fresh foreground validator run on that file shows 0 failures.** Edits can
   silently no-op (string-not-found) if the file drifted from a stale read; re-validate, don't assume.
4. The full rationale and the flaky-tool-channel defenses (trust `git hash-object`/refs over command stdout;
   write probe output in-repo not `/tmp`) live in the memory note `feedback_genjax_block_workflow.md`.

## Chapter quality bar — companion code, interactivity, and a clarity-review loop (do NOT skip before declaring a chapter done)

The code validator (above) proves a cell *runs*; it does not prove the chapter is *rich* or *clear*. The proven sequence (this is exactly how Ch 20–22 were built) is: draft → **iterate a student-persona clarity review to convergence** (item 3) → then a completeness pass for runnable GenJAX + interactivity (items 1–2) → then the every-time cross-ref checklist below. Three standing requirements before a chapter is done:

1. **Maximize runnable GenJAX companion code.** Interweave a runnable `@gen` GenJAX cell next to every concept that can carry one — this is the textbook's version of the lecture's "make as much as possible have GenJAX code." **Reuse the verified lecture backbones** (`course/weekNN_*/genjax_*.py`, already run against genjax 0.10.3) as the chapter's spine rather than re-deriving. Hard rule before completing: every code cell either **executes under the validator with a matching `**Output:**`**, or is *deliberately* marked `<!-- validate: skip -->` **with a one-line reason** — never leave a cell that silently can't run, and never ship a chapter whose GenJAX hasn't passed a fresh validator run. If a cell won't run, fix it or cut it; don't paper over it.

2. **Embed the interactivity.** The lecture widgets are dual-purpose (Week 8/9 pattern: a widget lives once and is referenced from both the deck and the chapter). Iframe the relevant widget(s) into the chapter, each with a static fallback image and a one-line "what to try." A chapter that has a matching lecture widget but doesn't embed it is incomplete.

3. **Student-agent clarity review, iterated to convergence** — *the proven Ch 20–22 loop; follow it, don't improvise.* After a chapter drafts, spawn a **student-persona `Explore` agent** to read it and critique, rewrite the chapter from the critique, then spawn the next round — **repeat until the feedback is minimal.** In practice that was **2–3 rounds** per chapter, stopping when the clarity rating reached **~8/10 AND every targeted fix from the prior round was confirmed resolved** — *not* "9/10 / zero issues," and *never* on a single pass. (Note: the textbook loop is **one diligent-student persona iterated across rounds**, distinct from the lecture deck's parallel-personas-single-pass; you may add a second-background persona on round 1 for breadth, but the core is iterate-to-convergence.)

   - **Persona template (fill the brackets per chapter):** *"You are a diligent student working through this probability / probabilistic-computing textbook. You have carefully studied the earlier chapters — [name the prerequisites, e.g. Markov chains, Monte Carlo, statistical decision theory] — so you are comfortable with [the carried-over concepts and notation, e.g. posteriors, $\mathbb{E}[\cdot]$, basic JAX]. But you are **brand new to [this chapter's topic]** — you've never seen [the specific new terms] before."* Always name the **read order** (this chapter + its prerequisites) so the agent can check every "you already saw this" claim.
   - **Review rubric — tell the agent to return exactly this, and to NOT edit files (critique as its final message):** (1) **confusing passages** — where did you get lost / have to re-read, and why (quote exact lines); (2) **undefined / under-defined terms or symbols, AND any place the code and the math disagree** (does the formula match what the cell computes? are code variable names consistent with the symbols? is one symbol overloaded — e.g. `θ` as both "state" and "probability"?); (3) **missing intuition or a figure that should exist** — especially the *why* (the canonical example: "why does 0–1 loss give the mode?"); (4) **notation / difficulty jumps**; (5) **what worked well** (so it's kept); (6) **comprehension check + verdict** — "could you now (a) state…, (b) write…, (c) *justify* (not just recall)…?" + a **clarity rating 1–10** + the top fixes still worth making. Add: "if the chapter is essentially clear with only trivial polish left, **say so explicitly**."
   - **Round 2+:** narrow the persona to *verify the prior round's fixes* — "confirm these are now resolved: (A)… (B)… (C)… — **Resolved / Partial / Not**" — and raise the bar from recall to justification.
   - **Textbook-specific checks the reviewer must do (beyond a slide review):** read the **built Hugo HTML**, not just the markdown, so MathJax rendering and layout are judged as a student sees them; **run every code cell and confirm its printed output matches the `**Output:**`** in the prose (the most common real defect was code↔prose drift); verify cross-chapter claims and glossary/anchor links resolve; and check **notation consistency between prose and code** (`R(s,a)` ↔ `RM[s][a]`).
   - **Record** the per-round ratings and any *known residual* non-trivial defect you consciously deferred (e.g. "summary box still contradicts the body, line 332"), so an 8/10 isn't silently shipped as a 10/10.

## Every-time checklist when a chapter is added or updated (Joe's standing rule, 2026-06-03)

Whenever you ship a **new chapter** or materially update one, do ALL of the following in the same pass —
not just the chapter file:

1. **Global notebook page.** Add the chapter's Colab notebook to `content/notebook_guide.md` in all three
   places that match the 08–11 pattern: the detailed section (Colab link + "What it covers" + chapter
   back-link + Topics), a recommended-learning-path entry, and the summary table. Verify chapter→notebook
   (in the chapter) and notebook→chapter (in the guide) links both resolve.
2. **Glossary.** Add the chapter's important new terms to `content/glossary.md` (the `### Term 📊` +
   `{{% expand %}}` format), each with an **"Appears in:"** link to the chapter section and **"See also:"**
   cross-links. Add a `*Glossary:*` line to the chapter's closing "What you can do now" box linking the terms
   back. **Verify every anchor in BOTH directions on the BUILT site** — Hugo keeps `π` (UTF-8) in anchors,
   drops apostrophes with no dash (`what's`→`whats`), and a trailing emoji becomes a trailing `-`
   (`markov-property-`). Don't guess slugs; grep `id="..."` in the built HTML.
3. **HML homepage lecture card.** In the COURSE repo (not the textbook), edit the week's
   `course/weekNN_*/PLAN.md` **"Textbook Chapters"** section to tag the chapters in the order-robust NAME form
   `- T3: <stable-name> — \`intro2/NN_slug.md\` (note)`, where `<stable-name>` is the slug minus its `NN_`
   prefix with `_`→spaces and **must match the slug exactly** (e.g. `random walks networks`, NOT "random walks
   *on* networks" — the matcher normalizes against the slug). Then run `python3 docs/_build.py` (never
   hand-edit `docs/index.html`) and confirm the three deep-links render on the week's card. Commit PLAN.md +
   the regenerated index.html to the course repo's main.

Bump `date` on every file you touch. The notebook-guide/glossary/index.html are easy to forget because they
live away from the chapter — this checklist exists because they were forgotten on the first Ch 13–15 pass.
