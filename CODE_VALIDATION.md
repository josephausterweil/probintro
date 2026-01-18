# Code Block Validation

This document describes the code validation workflow for tutorial markdown files.

## Overview

All Python/JAX code blocks in tutorial markdown files are automatically validated for:
- **Python syntax errors** - Ensures code will parse correctly
- **JAX compatibility** - Checks for common JAX usage patterns
- **Import completeness** - Verifies required imports are present

## Automated Validation

### Pre-Commit Hook

A Git pre-commit hook automatically validates all code blocks before allowing commits:

```bash
# Normal commit - validation runs automatically
git commit -m "Update tutorial"

# Bypass validation (not recommended)
git commit --no-verify -m "Update tutorial"
```

If validation fails, you'll see detailed error messages indicating which code blocks need fixing.

### CI/CD Integration

GitHub Actions automatically validates code blocks on:
- Pushes to `main` branch
- Pull requests to `main` branch
- Manual workflow dispatch

View validation status in the "Actions" tab of the GitHub repository.

## Manual Validation

Run validation manually anytime:

```bash
python validate_code_blocks.py
```

Output example:
```
Found 32 markdown files to validate

Validating content/genjax/02_first_model.md...
  Found 13 code blocks
  ✅ Block at line 27
  ✅ Block at line 44
  ...

============================================================
✅ All code blocks validated successfully
```

## Validation Rules

### Python Syntax

All code blocks must be valid Python syntax:

```python
# ✅ Valid
def hello():
    print("Hello, world!")

# ❌ Invalid - SyntaxError
def hello()
    print("Hello, world!")
```

### JAX Compatibility

#### Rule 1: Use `jnp.where()` for Conditionals

Python `if/else` statements don't work with JAX arrays:

```python
# ❌ Don't use Python if/else
if x > 0:
    result = x
else:
    result = -x

# ✅ Use jnp.where()
result = jnp.where(x > 0, x, -x)
```

#### Rule 2: Include Required Imports

All code blocks should be self-contained with necessary imports:

```python
# ❌ Missing imports
x = jnp.array([1, 2, 3])  # jnp not imported
key = jr.PRNGKey(0)       # jr not imported

# ✅ Complete imports
import jax.numpy as jnp
import jax.random as jr

x = jnp.array([1, 2, 3])
key = jr.PRNGKey(0)
```

#### Rule 3: GenJAX Decorators Require Imports

```python
# ❌ Missing GenJAX import
@genjax.gen
def my_model():
    ...

# ✅ Include import
from genjax import generative as genjax

@genjax.gen
def my_model():
    ...
```

## Writing New Tutorials

When adding new code blocks to tutorials:

1. **Write complete, runnable code** - Include all necessary imports
2. **Test locally** - Run `python validate_code_blocks.py` before committing
3. **Follow JAX patterns** - Use `jnp.where()` instead of `if/else`
4. **Mark code block language** - Use ` ```python ` or ` ```jax `

### Example Template

```python
# Import all required libraries
import jax.numpy as jnp
import jax.random as jr
from genjax import generative as genjax

# Define complete, runnable code
key = jr.PRNGKey(0)
data = jnp.array([1.0, 2.0, 3.0])

# Use JAX-compatible conditionals
result = jnp.where(data > 1.5, data, 0.0)

# Include any necessary function definitions
@genjax.gen
def my_model():
    x = jnp.normal() @ "x"
    return x
```

## Troubleshooting

### Validation Fails on Commit

If pre-commit validation fails:

1. **Read the error message** - It shows which file and line number has issues
2. **Fix the code block** - Address syntax errors or missing imports
3. **Test manually** - Run `python validate_code_blocks.py`
4. **Commit again** - Once validation passes, commit will succeed

### Common Issues

#### Missing Import

```
⚠️  Block at line 123:
    Missing import: 'import jax.numpy as jnp'
```

**Fix**: Add the import at the start of the code block.

#### Python if/else

```
⚠️  Block at line 456:
    Consider using jnp.where() instead of Python if/else
```

**Fix**: Replace `if/else` with `jnp.where()`:
```python
# Before
if condition:
    result = a
else:
    result = b

# After
result = jnp.where(condition, a, b)
```

## Maintenance

### Updating Validation Rules

To add new validation rules:

1. Edit `validate_code_blocks.py`
2. Add new check functions (following existing patterns)
3. Test on existing tutorials: `python validate_code_blocks.py`
4. Update this documentation with new rules

### Excluding Files

To exclude specific markdown files from validation:

Edit `validate_code_blocks.py` and modify the `tutorial_dirs` list:

```python
# Exclude specific directories
tutorial_dirs = ["intro", "intro2", "genjax"]  # Remove dirs you want to skip
```

## References

- **Validation Script**: `validate_code_blocks.py`
- **Pre-Commit Hook**: `.git/hooks/pre-commit`
- **CI/CD Workflow**: `.github/workflows/validate-code-blocks.yml`
- **GenJAX Guidelines**: `content/genjax/WRITING_GUIDE.md`