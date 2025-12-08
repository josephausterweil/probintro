# GenJAX Tutorial Writing Guidelines

**Purpose**: Prevent common errors when creating new GenJAX tutorial content

**Date Created**: 2025-12-08

---

## Critical Rules - Must Follow

### 1. ✅ ALWAYS Use Importance Sampling for Inference

When using `.generate()` for inference, **ALWAYS** use the weights for importance sampling.

#### ❌ WRONG - Returns Prior, Not Posterior
```python
def infer(k):
    trace, weight = model.generate(k, obs, args)
    return trace.get_retval()  # ❌ Discarding weight!

samples = jax.vmap(infer)(keys)
prob = jnp.mean(samples)  # ❌ Simple average = PRIOR
```

#### ✅ CORRECT - Returns Posterior
```python
def infer(k):
    trace, weight = model.generate(k, obs, args)
    return trace.get_retval(), weight  # ✅ Keep both!

results = jax.vmap(infer)(keys)
samples, weights = results[0], results[1]

# ✅ Use importance sampling
normalized_weights = jnp.exp(weights - jnp.max(weights))
normalized_weights = normalized_weights / jnp.sum(normalized_weights)
prob = jnp.sum(samples * normalized_weights)
```

**Why**: Simple averaging returns the prior probability. Importance sampling with weights returns the posterior probability.

**When**: This applies to ALL inference code using `.generate()` with observations.

---

### 2. ✅ NEVER Use Python if/else in @gen Functions

JAX tracing cannot handle Python control flow with traced values.

#### ❌ WRONG - Causes TracerBoolConversionError
```python
@gen
def model():
    is_blue = flip(0.5) @ "is_blue"

    # ❌ Python if/else with traced value
    if is_blue:
        value = normal(10.0, 1.0) @ "value"
    else:
        value = normal(5.0, 1.0) @ "value"

    return value
```

#### ✅ CORRECT - Use jnp.where()
```python
@gen
def model():
    is_blue = flip(0.5) @ "is_blue"

    # ✅ Use jnp.where for conditional logic
    mean = jnp.where(is_blue, 10.0, 5.0)
    value = normal(mean, 1.0) @ "value"

    return value
```

**Why**: JAX's tracing system transforms code, and traced values cannot be used in Python boolean contexts.

**When**: This applies to ALL @gen decorated functions that use conditional logic based on random choices.

**Also Applies To**: Python `and`, `or`, `not`, `while`, `for` with traced conditions

---

### 3. ✅ ALWAYS Use ChoiceMap.d() Constructor

ChoiceMap requires the `.d()` method for dictionary construction.

#### ❌ WRONG - Causes "Can't instantiate abstract Struct" Error
```python
observation = ChoiceMap({"says_blue": True})  # ❌
```

#### ✅ CORRECT - Use .d() Method
```python
observation = ChoiceMap.d({"says_blue": True})  # ✅
```

**Why**: ChoiceMap is an abstract class that requires the `.d()` factory method.

**When**: Every time you create a ChoiceMap from a dictionary.

---

### 4. ✅ ALWAYS Make Code Blocks Independently Runnable

Every code block in a tutorial must be able to run standalone.

#### ❌ WRONG - Missing Imports and Variables
```python
@gen
def model():
    value = normal(mu, sigma) @ "value"  # ❌ mu, sigma undefined
    return value

key = random.PRNGKey(42)  # ❌ random not imported
```

#### ✅ CORRECT - Complete, Runnable Code
```python
import jax
import jax.numpy as jnp
from genjax import gen, simulate
import jax.random as random

# Define any needed parameters
mu = 500.0
sigma = 2.0

@gen
def model():
    value = normal(mu, sigma) @ "value"
    return value

key = random.PRNGKey(42)
```

**Why**: Students copy code blocks to try them. If they don't run, it's frustrating and wastes learning time.

**When**: Every code block, even if it seems obvious from context.

**Required Elements**:
- All imports at the top
- All variable/data definitions
- All function definitions
- Key initialization if using JAX random

---

## Code Review Checklist

Before committing new tutorial content, verify:

### For Inference Code:
- [ ] Does the code use `.generate()` with observations?
- [ ] Are the weights from `.generate()` being used?
- [ ] Is importance sampling applied (normalize weights, weighted sum)?
- [ ] Are results validated against analytical solutions (if available)?

### For @gen Functions:
- [ ] Does the function use any conditional logic?
- [ ] Are all conditionals using `jnp.where()` (not `if/else`)?
- [ ] Are there any `if`, `while`, `for` statements with traced conditions?
- [ ] Does the function compile without TracerBoolConversionError?

### For ChoiceMap Usage:
- [ ] Are all ChoiceMaps constructed with `.d()` method?
- [ ] Test: Can you run `ChoiceMap.d({...})` without error?

### For Code Blocks:
- [ ] Does the code block include all necessary imports?
- [ ] Are all variables and data used in the block defined?
- [ ] Can you copy-paste the block into a fresh Python file and run it?
- [ ] Does it run without `NameError`, `ImportError`, or `AttributeError`?

---

## Common Patterns - Quick Reference

### Pattern 1: Conditional Distribution in @gen
```python
@gen
def conditional_model():
    # Sample discrete choice
    category = categorical(probs) @ "category"

    # ✅ Use jnp.where to select parameters
    mean = jnp.where(category == 0, mu_0, mu_1)
    std = jnp.where(category == 0, sigma_0, sigma_1)

    # Sample from conditional distribution
    value = normal(mean, std) @ "value"
    return value
```

### Pattern 2: Multiple Conditions
```python
@gen
def multi_conditional():
    x = flip(0.5) @ "x"
    y = flip(0.5) @ "y"

    # ✅ Nest jnp.where for multiple conditions
    # Equivalent to: if x and y: return 4; elif x: return 3; elif y: return 2; else: return 1
    value = jnp.where(x,
                      jnp.where(y, 4.0, 3.0),  # if x
                      jnp.where(y, 2.0, 1.0))  # if not x

    result = normal(value, 1.0) @ "result"
    return result
```

### Pattern 3: Complete Inference Example
```python
import jax
import jax.numpy as jnp
from genjax import gen, flip, ChoiceMap
import jax.random as random

@gen
def model(accuracy):
    is_blue = flip(0.15) @ "is_blue"
    says_blue_prob = jnp.where(is_blue, accuracy, 1 - accuracy)
    says_blue = flip(says_blue_prob) @ "says_blue"
    return is_blue

# Create observation
observation = ChoiceMap.d({"says_blue": True})

# Inference with importance sampling
key = random.PRNGKey(42)
keys = jax.random.split(key, 10000)

def infer(k):
    trace, weight = model.generate(k, observation, (0.80,))
    return trace.get_retval(), weight

results = jax.vmap(infer)(keys)
samples, weights = results[0], results[1]

# Importance sampling
normalized_weights = jnp.exp(weights - jnp.max(weights))
normalized_weights = normalized_weights / jnp.sum(normalized_weights)
prob_blue = jnp.sum(samples * normalized_weights)

print(f"P(Blue | says Blue) ≈ {prob_blue:.3f}")
```

---

## Testing New Content

### Automated Checks

Run these before committing:

```bash
# Check for Python if/else in @gen functions
grep -n "if " content/**/*.md | grep -B5 "@gen"

# Check for ChoiceMap without .d()
grep -n "ChoiceMap({" content/**/*.md

# Check for .generate() without weight usage
grep -n "\.generate(" content/**/*.md
```

### Manual Verification

For each new tutorial:

1. **Extract all code blocks** to separate Python files
2. **Run each code block** independently
3. **Verify outputs** match expected results
4. **Test with fresh Python environment** (no assumed imports)

### Validation Against Analytics

For inference examples:

1. **Compute analytical solution** using Bayes' theorem
2. **Compare simulation results** to analytical
3. **Verify convergence** with different sample sizes
4. **Document expected values** in tutorial

---

## Why These Guidelines Matter

### Student Experience
- **Correct inference**: Students learn proper Bayesian methods
- **Runnable code**: Students can experiment immediately
- **No frustrating errors**: Clear examples that work first time

### Tutorial Quality
- **Mathematically correct**: Posteriors match theory
- **Technically correct**: JAX-compatible, no runtime errors
- **Practically usable**: Copy-paste-run workflow

### Maintenance
- **Fewer bug reports**: Code works as written
- **Easier updates**: Clear patterns to follow
- **Self-documenting**: Examples demonstrate best practices

---

## Common Mistakes to Avoid

### Mistake 1: "I tested it and it worked"
- ❌ Testing in a notebook with prior cells run
- ✅ Testing by copy-pasting each block into fresh Python file

### Mistake 2: "The imports are obvious"
- ❌ Assuming students know what to import
- ✅ Including all imports in every code block

### Mistake 3: "Simple average is close enough"
- ❌ Using `jnp.mean()` for inference (returns prior)
- ✅ Always using importance sampling (returns posterior)

### Mistake 4: "Python if/else is more readable"
- ❌ Using Python control flow in @gen functions
- ✅ Using `jnp.where()` (JAX-compatible)

### Mistake 5: "ChoiceMap() worked for me"
- ❌ Depending on environment-specific behavior
- ✅ Using documented `ChoiceMap.d()` API

---

## When in Doubt

### Questions to Ask:

1. **Can a student copy-paste this code and run it immediately?**
   - If no: Add missing imports/definitions

2. **Does this code use `.generate()` with observations?**
   - If yes: Verify importance sampling is used

3. **Does this @gen function have conditional logic?**
   - If yes: Verify it uses `jnp.where()`, not `if/else`

4. **Does this code create a ChoiceMap?**
   - If yes: Verify it uses `.d()` method

5. **Have I tested this code in a fresh environment?**
   - If no: Test it before committing

### Resources:

- **This document**: Complete guidelines
- **Example tutorials**: `content/genjax/04_conditioning.md` (after fixes)
- **Test files**: `test_conditioning.py`, `test_building_models.py`
- **Fix summaries**: `FINAL_COMPLETE_SUMMARY.md`

---

## Template for New Tutorial Sections

Use this template when writing new content:

```markdown
## [Your Topic]

[Explanation text]

```python
# ✅ TEMPLATE: Complete, runnable code block

# 1. All necessary imports
import jax
import jax.numpy as jnp
from genjax import gen, flip, normal, ChoiceMap, simulate
import jax.random as random

# 2. Define any needed data/parameters
param_1 = 0.5
data = jnp.array([...])

# 3. Define model with JAX-compatible conditionals
@gen
def model():
    choice = flip(param_1) @ "choice"
    # ✅ Use jnp.where, not if/else
    value = jnp.where(choice, 1.0, 0.0)
    return value

# 4. Set up inference if applicable
observation = ChoiceMap.d({"choice": True})  # ✅ Use .d()

key = random.PRNGKey(42)
keys = jax.random.split(key, 1000)

# 5. Inference with importance sampling (if applicable)
def infer(k):
    trace, weight = model.generate(k, observation, ())
    return trace.get_retval(), weight  # ✅ Keep weight!

results = jax.vmap(infer)(keys)
samples, weights = results[0], results[1]

# ✅ Use importance sampling
normalized_weights = jnp.exp(weights - jnp.max(weights))
normalized_weights = normalized_weights / jnp.sum(normalized_weights)
result = jnp.sum(samples * normalized_weights)

print(f"Result: {result:.3f}")
\```

**Expected Output:**
\```
Result: [expected value]
\```

[Explanation of results]
```

---

## Enforcement

### Before Commit:
- [ ] Run automated checks
- [ ] Test all code blocks independently
- [ ] Verify inference uses importance sampling
- [ ] Verify @gen functions use jnp.where()
- [ ] Verify ChoiceMap uses .d()

### Code Review:
- Reviewer must verify all guidelines followed
- No exceptions without explicit justification
- Failed checks = revision required

### Continuous Improvement:
- Update this document as new patterns emerge
- Add examples from real mistakes
- Keep guidelines current with GenJAX updates

---

**Remember**: These guidelines exist because we made these mistakes. Following them prevents repeating them.

**Last Updated**: 2025-12-08
**Based On**: Fixes to 33 issues across 5 tutorials
**Next Review**: When GenJAX API changes or new patterns emerge
