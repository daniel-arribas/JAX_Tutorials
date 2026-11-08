# 4: JIT & vmap

This module covers how to eliminate Python loops and maximize hardware execution speed using JAX's compilation and vectorization engines.

## 1. `jax.vmap` (Vectorized Map)
`vmap` automatically transforms a function that operates on a single scalar sample into a function that processes an entire batch/vector of data seamlessly. It pushes the looping logic down to highly optimized low-level operations.

## 2. `jax.jit` (Just-In-Time Compilation)
`jit` takes a standard Python/JAX function, compiles its operations into a fused execution graph using the **XLA (Accelerated Linear Algebra)** compiler, and runs it at native hardware speed on the CPU. The first execution has a small compilation overhead, but subsequent runs are blindingly fast.

### Combination Pattern
By wrapping a vectorized function inside JIT (`jax.jit(jax.vmap(f))`), you achieve massive parallel efficiency even on standard consumer CPUs.
