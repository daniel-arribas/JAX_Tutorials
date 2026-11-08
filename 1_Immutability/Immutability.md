markdown
- # 1: Immutability
- ## Concept: Functional Purity & Array Immutability
	- Unlike NumPy or PyTorch arrays, JAX arrays (**DeviceArrays**) are strictly **immutable**.
	- **Why?** JAX relies on the XLA compiler to build optimized execution graphs. In-place mutations (like `x = 5`) break these graph assumptions and introduce side effects.
	- JAX functions must be **pure**: they must not modify global states or input variables in-place.
- ## The JAX Solution: `.at[index].set(value)`
	- JAX provides a functional indexing syntax under the `.at` property.
	- Executing `y = x.at.set(5)` does **not** modify `x`. Instead, XLA optimizes this operation behind the scenes to efficiently allocate and return a *new* array `y` with the updated value.
- ## Key Code Snippet
	```python
	import jax.numpy as jnp
	x = jnp.array()
	# y becomes , while x stays 
	y = x.at.set(99)
	```