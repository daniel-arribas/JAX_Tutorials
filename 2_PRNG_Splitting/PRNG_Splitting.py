"""
PRNG_Splitting.py

In this tutorial, we will explore JAX's Stateless Pseudo-Random Number Generation (PRNG) 
and understand why explicit key splitting is mandatory.

We first import the required dependencies

"""

import jax
import jax.numpy as jnp

"""
---------------------------------------------------------------------------------------------------------------------------
We initialize the PRNG key

"""

# 1. Initialize the baseline PRNG key (Deterministic seed)
base_key = jax.random.PRNGKey(2026)
print(f"📌 Initial JAX Key Object: {base_key}")

"""
----------------------------------------------------------------------------------------------------------------------
First, we demonstrate the side-effect of reusing the exact same key.
Because JAX functions are pure, the same key input will yield identical outputs.
"""
print("\n--- ❌ Reusing the exact same key ---")

random_vector_1 = jax.random.uniform(base_key, shape=(2,))
random_vector_2 = jax.random.uniform(base_key, shape=(2,))

print(f"First call output:  {random_vector_1}")
print(f"Second call output: {random_vector_2}")
print(f"Are both vectors identical? {jnp.array_equal(random_vector_1, random_vector_2)}")

"""
----------------------------------------------------------------------------------------------------------------------
To fix this, we must explicitly split the key. 
jax.random.split() forks a key into a new operational key and an ephemeral subkey.
"""
print("\n--- ✅ Splitting the key properly ---")

# We split the baseline key into a new tracking key and a subkey for immediate consumption
new_key, subkey_1 = jax.random.split(base_key)
# We split again from the recently updated tracking key
new_key, subkey_2 = jax.random.split(new_key)

random_vector_3 = jax.random.uniform(subkey_1, shape=(2,))
random_vector_4 = jax.random.uniform(subkey_2, shape=(2,))

print(f"First subkey output:  {random_vector_3}")
print(f"Second subkey output: {random_vector_4}")
print(f"Are they identical now? {jnp.array_equal(random_vector_3, random_vector_4)}")

"""
----------------------------------------------------------------------------------------------------------------------
Pro-Tip: If you need multiple keys at once (e.g., for initializing weights in different network layers), 
you can pass an integer to the split function to get a batch of subkeys instantly.
"""
print("\n--- ⚡ Generating a batch of subkeys ---")

# Generate 3 subkeys at once
new_key, *subkeys = jax.random.split(new_key, num=4) # Returns 1 master key + 3 subkeys packed in a list

print(f"Number of generated subkeys: {len(subkeys)}")
for i, sk in enumerate(subkeys):
    print(f"Subkey {i}: {sk} -> Random Value: {jax.random.normal(sk, shape=(1,))}")
