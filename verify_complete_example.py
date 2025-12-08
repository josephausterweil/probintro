#!/usr/bin/env python3
"""
Verify the complete example from genjax/04_conditioning tutorial.
This extracts the exact code from the tutorial and tests it.
"""

# This code is extracted directly from content/genjax/04_conditioning.md
# Lines 655-695 (the "Complete Example" section)

try:
    import jax
    import jax.numpy as jnp
    from genjax import gen, flip, ChoiceMap

    print("✅ Dependencies imported successfully")
    print(f"JAX version: {jax.__version__}")

except ImportError as e:
    print("❌ Missing dependencies!")
    print(f"Error: {e}")
    print("\nTo run this test, you need:")
    print("  pip install jax jaxlib")
    print("  pip install genjax")
    exit(1)

@gen
def taxicab_model(base_rate_blue=0.15, accuracy=0.80):
    """Taxicab problem generative model."""
    is_blue = flip(base_rate_blue) @ "is_blue"

    # What Chibany says depends on true color
    says_blue_prob = jnp.where(is_blue, accuracy, 1 - accuracy)
    says_blue = flip(says_blue_prob) @ "says_blue"

    return is_blue

# Observation: Chibany says "blue"
observation = ChoiceMap.d({"says_blue": True})

# Generate posterior samples
key = jax.random.key(42)
keys = jax.random.split(key, 10000)

def run_inference(k):
    trace, weight = taxicab_model.generate(k, observation, (0.15, 0.80))
    return trace.get_retval(), weight

results = jax.vmap(run_inference)(keys)
is_blue_samples = results[0]
weights = results[1]

# Use importance sampling with weights
normalized_weights = jnp.exp(weights - jnp.max(weights))
normalized_weights = normalized_weights / jnp.sum(normalized_weights)
prob_blue = jnp.sum(is_blue_samples * normalized_weights)

print(f"\n=== TAXICAB INFERENCE ===")
print(f"Base rate: 15% blue")
print(f"Accuracy: 80%")
print(f"Observation: Says 'blue'")
print(f"\nP(Blue | says Blue) ≈ {prob_blue:.3f}")

# Verify against analytical solution
expected = 0.414
diff = abs(prob_blue - expected)

print(f"\n=== VERIFICATION ===")
print(f"Expected (Bayes' theorem): {expected:.3f}")
print(f"Got (importance sampling): {prob_blue:.3f}")
print(f"Difference: {diff:.3f}")

if diff < 0.02:
    print("\n✅ SUCCESS! Code produces correct result")
    print("   Tutorial example is working correctly")
else:
    print(f"\n❌ FAILURE! Result differs by {diff:.3f}")
    print("   Expected ≈0.414, tutorial might still have issues")
