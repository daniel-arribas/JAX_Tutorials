# 2: Pseudo-Random Number Generation (PRNG)

In standard data science libraries like NumPy or PyTorch, random number generation is **stateful**. Every time you call a random function, an implicit global seed updates in the background. 

**JAX operates differently:** to enforce functional purity, reproducibility, and prevent race conditions when running operations across parallel cores, JAX uses a **stateless PRNG system**.

## The Core Concept: Stateless Randomness

JAX random functions are mathematically **pure**. This means that passing the exact same `PRNGKey` into a random function will *always* generate the **identical random values**.

### Key Forking / Splitting Strategy

To generate a sequence of distinct random vectors, you must manually clone and update the state of your keys. JAX provides the `jax.random.split()` utility to accomplish this:

1. **`new_key`**: A new tracking state intended to be saved or passed down for the *next* split operation.
2. **`subkey`**: An ephemeral, one-time key consumed immediately by your target random function (e.g., initializing weights).

### Quick Code Reference

```python
import jax

# Initialize the main key
key = jax.random.PRNGKey(42)

# Explicitly fork the key before generating values
key, subkey = jax.random.split(key)
random_data = jax.random.normal(subkey, (3, 3))
```
