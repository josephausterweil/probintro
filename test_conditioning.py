#!/usr/bin/env python3
"""Test the taxicab conditioning code to diagnose the probability issue."""

import jax
import jax.numpy as jnp
from genjax import gen, flip, ChoiceMap

@gen
def taxicab_model(base_rate_blue=0.15, accuracy=0.80):
    """Generate the taxi color and what Chibany says.

    Args:
        base_rate_blue: Probability a taxi is blue (default 0.15)
        accuracy: Probability Chibany identifies correctly (default 0.80)

    Returns:
        True if taxi is blue, False if green
    """

    # True taxi color (blue = 1, green = 0)
    is_blue = flip(base_rate_blue) @ "is_blue"

    # What Chibany says depends on the true color
    # Use jnp.where for JAX compatibility
    says_blue_prob = jnp.where(is_blue, accuracy, 1 - accuracy)
    says_blue = flip(says_blue_prob) @ "says_blue"

    return is_blue


def test_conditioning():
    """Test the conditional sampling."""
    print("=== Testing Taxicab Conditioning ===\n")

    # Observation: Chibany says "blue"
    observation = ChoiceMap.d({"says_blue": 1})

    print(f"Observation: {observation}")
    print(f"Conditioning on: says_blue = 1 (True)\n")

    # Generate 10,000 traces conditional on observation
    key = jax.random.key(42)
    keys = jax.random.split(key, 10000)

    def run_conditional(k):
        trace, weight = taxicab_model.generate(k, observation, (0.15, 0.80))
        return trace.get_retval()

    posterior_samples = jax.vmap(run_conditional)(keys)

    # Calculate posterior probability
    prob_blue_posterior = jnp.mean(posterior_samples)
    print(f"P(Blue | says Blue) ≈ {prob_blue_posterior:.3f}")
    print(f"Expected: ~0.414\n")

    # Check if the issue is with boolean vs int
    print("=== Diagnostic: Checking boolean values ===")
    print(f"Type of posterior_samples[0]: {type(posterior_samples[0])}")
    print(f"First 10 samples: {posterior_samples[:10]}")
    print(f"Sum of True values: {jnp.sum(posterior_samples)}")
    print(f"Mean (as probability): {jnp.mean(posterior_samples)}")

    # Theoretical calculation
    print("\n=== Bayes' Theorem Verification ===")
    P_blue = 0.15
    P_green = 0.85
    P_says_blue_given_blue = 0.80
    P_says_blue_given_green = 0.20

    P_says_blue = (P_blue * P_says_blue_given_blue +
                   P_green * P_says_blue_given_green)

    P_blue_given_says_blue = (P_says_blue_given_blue * P_blue) / P_says_blue

    print(f"P(Blue) = {P_blue}")
    print(f"P(says Blue | Blue) = {P_says_blue_given_blue}")
    print(f"P(says Blue | Green) = {P_says_blue_given_green}")
    print(f"P(says Blue) = {P_says_blue:.3f}")
    print(f"P(Blue | says Blue) = {P_blue_given_says_blue:.3f}")

    # Check if result matches
    diff = abs(prob_blue_posterior - P_blue_given_says_blue)
    print(f"\nDifference: {diff:.3f}")
    if diff < 0.01:
        print("✅ Results match! (within 0.01)")
    else:
        print("❌ Results don't match!")
        print(f"   Got {prob_blue_posterior:.3f}, expected {P_blue_given_says_blue:.3f}")


if __name__ == "__main__":
    test_conditioning()
