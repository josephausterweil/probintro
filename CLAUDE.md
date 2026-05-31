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
