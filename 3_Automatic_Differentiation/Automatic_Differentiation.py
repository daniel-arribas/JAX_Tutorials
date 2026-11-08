"""
Automatic_Differentiation.py

In this tutorial, we will explore JAX's functional differentiation engine (`grad`) 
and demonstrate how to compute exact gradients by transforming functions on CPU.

"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# 1. Define a pure mathematical function: f(x) = 3x^2 + 2x + 1
def f(x):
    return 3.0 * (x ** 2) + 2.0 * x + 1.0

"""
----------------------------------------------------------------------------------------------------------------------
JAX Transformation: jax.grad()
Instead of calculating a tensor, jax.grad takes the function 'f' and generates a new function 'f_prime'.
Mathematically, the derivative is: f'(x) = 6x + 2
"""

f_prime = jax.grad(f)

# Evaluate the newly generated derivative function at x = 4.0
# Expected theoretical result: f'(4) = 6(4) + 2 = 26
x_target = 4.0
grad_value = f_prime(x_target)

print("--- 🧮 Functional Differentiation Evaluation ---")
print(f"Original Function value at f({x_target}) = {f(x_target)}")
print(f"JAX Calculated Gradient value f'({x_target}) = {grad_value}")
print(f"Is the JAX gradient exact (26.0)? {grad_value == 26.0}")


"""
----------------------------------------------------------------------------------------------------------------------
Optimization with jax.value_and_grad()
In Machine Learning training loops, you need both the loss value (to monitor it) and the gradients (to update weights).
Running jax.grad and then f(x) separately does double the work. value_and_grad solves this in a single forward/backward pass.
"""
print("\n--- ⚡ Double efficiency with jax.value_and_grad ---")

value_and_grad_fn = jax.value_and_grad(f)

# Evaluate both at once
loss_value, gradients = value_and_grad_fn(x_target)
print(f"Extracted Value:    {loss_value}")
print(f"Extracted Gradient: {gradients}")


"""
----------------------------------------------------------------------------------------------------------------------
Visual Verification using Matplotlib
We will plot the original function curve alongside its derivative line across a range.
"""
print("\n--- 📊 Plotting results to verify curves ---")

x_range = jnp.linspace(-5.0, 5.0, 100)
y_values = x_range**2 * 3.0 + x_range * 2.0 + 1.0 # Standard mathematical evaluation

# Note: We use a list comprehension here because jax.grad expects a single scalar input, 
# not an entire array vector. (In the next tutorial we will learn 'vmap' to avoid this loop!)
y_prime_values = [f_prime(x) for x in x_range]

plt.figure(figsize=(8, 4))
plt.plot(x_range, y_values, label="f(x) = 3x² + 2x + 1", color="blue")
plt.plot(x_range, y_prime_values, label="f'(x) = 6x + 2", color="orange", linestyle="--")
plt.axvline(x=x_target, color="red", alpha=0.5, linestyle=":", label=f"Evaluated Point (x={x_target})")

plt.title("JAX Functional Automatic Differentiation (CPU)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
print("Showing plot window...")
plt.show()
