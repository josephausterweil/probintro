#!/usr/bin/env python3
"""
Generate visualization images for GenJAX tutorial chapters.

This script runs the visualization code from the tutorial and saves the output as images.
Run from the project root directory:
    python scripts/generate_genjax_visualizations.py
"""

import jax
import jax.numpy as jnp
from genjax import gen, flip
import matplotlib.pyplot as plt
from pathlib import Path

# Output directory - Hugo serves static files from static/
OUTPUT_DIR = Path("static/images/genjax")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Generating GenJAX tutorial visualizations...")
print(f"Output directory: {OUTPUT_DIR}")

# ============================================================================
# Chapter 2: Your First GenJAX Model
# ============================================================================

print("\n1. Generating first_model_outcome_distribution.png...")


@gen
def chibany_day():
    """Generate one day of Chibany's meals."""
    lunch_is_tonkatsu = flip(0.5) @ "lunch"
    dinner_is_tonkatsu = flip(0.5) @ "dinner"
    return (lunch_is_tonkatsu, dinner_is_tonkatsu)


# Generate 10,000 days
key = jax.random.key(42)
keys = jax.random.split(key, 10000)


def run_one_day(k):
    trace = chibany_day.simulate(k, ())
    lunch, dinner = trace.get_retval()
    return jnp.array([lunch, dinner])


days = jax.vmap(run_one_day)(keys)

# Count each outcome
HH = jnp.sum((days[:, 0] == 0) & (days[:, 1] == 0))
HT = jnp.sum((days[:, 0] == 0) & (days[:, 1] == 1))
TH = jnp.sum((days[:, 0] == 1) & (days[:, 1] == 0))
TT = jnp.sum((days[:, 0] == 1) & (days[:, 1] == 1))

# Create bar chart
outcomes = ['HH', 'HT', 'TH', 'TT']
counts = [int(HH), int(HT), int(TH), int(TT)]

plt.figure(figsize=(8, 5))
plt.bar(outcomes, counts, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#f7dc6f'])
plt.xlabel('Outcome')
plt.ylabel('Count (out of 10,000)')
plt.title("Chibany's Meals: 10,000 Simulated Days")
plt.axhline(y=2500, color='gray', linestyle='--', label='Expected (2500 each)')
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'first_model_outcome_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Saved first_model_outcome_distribution.png")

# ============================================================================
# Chapter 4: Conditioning and Inference
# ============================================================================

print("\n2. Generating taxicab_prior_posterior.png...")


@gen
def taxicab_model(base_rate_blue=0.15, accuracy=0.80):
    """Taxicab problem generative model."""
    is_blue = flip(base_rate_blue) @ "is_blue"
    says_blue_prob = jnp.where(is_blue, accuracy, 1 - accuracy)
    says_blue = flip(says_blue_prob) @ "says_blue"
    return is_blue


# Run inference
from genjax import ChoiceMap

observation = ChoiceMap.d({"says_blue": 1})
key = jax.random.key(42)
keys = jax.random.split(key, 10000)


def run_inference(k):
    trace, weight = taxicab_model.generate(k, observation, (0.15, 0.80))
    return trace.get_retval()


posterior_samples = jax.vmap(run_inference)(keys)
prob_blue_posterior = float(jnp.mean(posterior_samples))

# Prior vs Posterior visualization
prior_blue = 0.15
prior_green = 0.85
posterior_blue = prob_blue_posterior
posterior_green = 1 - posterior_blue

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

categories = ['Green', 'Blue']
colors = ['#4ecdc4', '#6c5ce7']

# Prior
ax1.bar(categories, [prior_green, prior_blue], color=colors)
ax1.set_ylabel('Probability', fontsize=12)
ax1.set_title('Prior: Before Chibany Speaks', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 1)
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

for i, prob in enumerate([prior_green, prior_blue]):
    ax1.text(i, prob + 0.05, f'{prob:.1%}', ha='center', fontsize=14, fontweight='bold')

# Posterior
ax2.bar(categories, [posterior_green, posterior_blue], color=colors)
ax2.set_ylabel('Probability', fontsize=12)
ax2.set_title('Posterior: After Chibany Says "Blue"', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 1)
ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

for i, prob in enumerate([posterior_green, posterior_blue]):
    ax2.text(i, prob + 0.05, f'{prob:.1%}', ha='center', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'taxicab_prior_posterior.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Saved taxicab_prior_posterior.png")

print("\n3. Generating taxicab_base_rate_effect.png...")

# Test different base rates
base_rates = jnp.linspace(0.01, 0.99, 50)
posteriors = []

for rate in base_rates:
    def run_with_rate(k):
        trace, weight = taxicab_model.generate(k, observation, (float(rate), 0.80))
        return trace.get_retval()

    keys = jax.random.split(key, 1000)
    post = jax.vmap(run_with_rate)(keys)
    posteriors.append(float(jnp.mean(post)))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(base_rates, posteriors, linewidth=2, color='#6c5ce7')
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
plt.axvline(x=0.15, color='red', linestyle='--', alpha=0.7, label='Original problem (15%)')
plt.scatter([0.15], [0.414], color='red', s=100, zorder=5)

plt.xlabel('Base Rate: P(Blue)', fontsize=12)
plt.ylabel('Posterior: P(Blue | says Blue)', fontsize=12)
plt.title('How Base Rates Affect Inference\n(Chibany 80% accurate)', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'taxicab_base_rate_effect.png', dpi=150, bbox_inches='tight')
plt.close()
print("   ✓ Saved taxicab_base_rate_effect.png")

print("\n" + "="*60)
print("✅ All visualizations generated successfully!")
print(f"📁 Images saved to: {OUTPUT_DIR.absolute()}")
print("="*60)
