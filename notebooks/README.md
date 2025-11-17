# GenJAX Tutorial Notebooks

## 02_first_genjax_model.ipynb

This notebook demonstrates GenJAX generative functions using Chibany's meal example.

### Important Note: Use flip(), not bernoulli()

The notebook uses GenJAX's `flip()` function for Bernoulli distributions, not `bernoulli()`. This is because:

**GenJAX has a bug in versions 0.9.3 and 0.10.3** where the `bernoulli(p)` function ignores the probability parameter and produces incorrect results.

**Test Results:**
- `bernoulli(0.9)` produces ~71% instead of 90% ❌
- `flip(0.9)` produces ~90% as expected ✅

**Solution:** Always use `flip(p)` for Bernoulli distributions in GenJAX. This is what the official GenJAX examples use.

### Environment Setup

The notebook requires:
- Python 3.12+
- JAX 0.5.3 with CUDA 12 support (for GPU acceleration)
- GenJAX 0.10.3
- ipywidgets, matplotlib, ipykernel

Install dependencies:
```bash
cd /home/jausterw/work/tutorials/amplifier_play
pip install -r requirements.txt
```

Or use the virtual environment:
```bash
source /home/jausterw/work/tutorials/amplifier_play/.venv/bin/activate
```

### GPU Support

The notebook is configured to run on NVIDIA GPUs with CUDA 12. If you have a compatible GPU, JAX will automatically use it. You can verify with:

```python
import jax
print(jax.devices())  # Should show CudaDevice(id=0)
```

### Historical Note

This notebook was corrected on 2025-11-17 to use `flip()` instead of `bernoulli()` after discovering the bug. The original notebook used `bernoulli()` which caused simulations to not match theoretical probabilities.

A backup of the original (broken) version is saved as `02_first_genjax_model.ipynb.backup` for reference.
