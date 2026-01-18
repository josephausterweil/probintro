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
