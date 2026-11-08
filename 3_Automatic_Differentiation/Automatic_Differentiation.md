# Phase 2: The 4 Grand Transformations - Automatic Differentiation (`grad`)

In standard Deep Learning frameworks like PyTorch or TensorFlow, gradients are computed using a **graph-based/variable-based** approach (e.g., calling `.backward()` on a loss tensor). 

**JAX introduces a functional paradigm:** it treats differentiation as a **functional transformation**. You pass a Python function `f(x)` into `jax.grad()`, and it returns a completely new Python function `f'(x)` representing its exact analytical derivative.

## Core Transformation Tools

1. **`jax.grad(f)`**: Computes the gradient of a scalar-valued function `f` with respect to its first argument (`argnums=0` by default).
2. **`jax.value_and_grad(f)`**: Efficiently computes both the function's output value and its gradient in a single mathematical pass. This is the gold standard for training loops.

### Rules for `grad`
- The target function must be a **pure function** (no side-effects, no mutable global state).
- The output of the function must be a **scalar** (a single float value, not a matrix/vector).

### Quick Code Reference

```python
import jax

# Transform the function
grad_fn = jax.grad(my_loss_function)

# Evaluate the derivative function like any normal python function
gradients = grad_fn(parameters)
```
