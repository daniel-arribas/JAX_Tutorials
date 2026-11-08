"""
JIT_and_VMAP.py

In this tutorial, we will benchmark how `jax.vmap` removes Python loops and 
how `jax.jit` uses the XLA compiler to optimize execution speeds on CPU.
"""

import jax
import jax.numpy as jnp
import time

# 1. Define our scalar derivative function from the previous tutorial
def f(x):
    return 3.0 * (x ** 2) + 2.0 * x + 1.0

f_prime = jax.grad(f)

# 2. Generate a large vector of 50,000 data points to stress test the CPU
large_vector = jnp.linspace(-10.0, 10.0, 50000)

print(f"📊 Benchmarking with a dataset of {len(large_vector)} elements...\n")

"""
----------------------------------------------------------------------------------------------------------------------
APPROACH 1: The Native Python Loop (Slow)
We evaluate the gradient element by element using standard list comprehension.
"""
start_time = time.time()
loop_results = [f_prime(x) for x in large_vector]
end_time = time.time()
loop_duration = end_time - start_time
print(f"🐢 Approach 1 (Python Loop): {loop_duration:.5f} seconds")

"""
----------------------------------------------------------------------------------------------------------------------
APPROACH 2: Automatic Vectorization with jax.vmap
We transform 'f_prime' so it accepts vectors natively. No python loops.
"""
# jax.vmap(f_prime) generates a brand new function that handles arrays natively
vectorized_f_prime = jax.vmap(f_prime)

start_time = time.time()
vmap_results = vectorized_f_prime(large_vector)
# JAX uses asynchronous dispatch, block_until_ready() ensures exact time tracking
vmap_results.block_until_ready() 
end_time = time.time()
vmap_duration = end_time - start_time
print(f"🚀 Approach 2 (jax.vmap):    {vmap_duration:.5f} seconds")

"""
----------------------------------------------------------------------------------------------------------------------
APPROACH 3: Fused Compilation with jax.jit + jax.vmap
We compile the vectorized function using the XLA compiler to make it run at native machine speed.
"""
# We apply jit to the already vectorized function
jitted_vectorized_f_prime = jax.jit(vectorized_f_prime)

# Run 1: Compilation Warm-up (XLA builds the graph here)
start_time = time.time()
jitted_results_warmup = jitted_vectorized_f_prime(large_vector)
jitted_results_warmup.block_until_ready()
end_time = time.time()
jit_warmup_duration = end_time - start_time
print(f"⚡ Approach 3 (JIT Warm-up): {jit_warmup_duration:.5f} seconds (Compiling graph...)")

# Run 2: Native Execution (Blindingly fast)
start_time = time.time()
jitted_results_fast = jitted_vectorized_f_prime(large_vector)
jitted_results_fast.block_until_ready()
end_time = time.time()
jit_fast_duration = end_time - start_time
print(f"🔥 Approach 3 (JIT Cached):  {jit_fast_duration:.5f} seconds")

"""
----------------------------------------------------------------------------------------------------------------------
Verification & Final Metric
"""
acceleration_factor = loop_duration / jit_fast_duration
print(f"\n📈 Final Verdict: JIT + VMAP is roughly {acceleration_factor:.1f}x faster than the standard Python loop on your CPU!")
