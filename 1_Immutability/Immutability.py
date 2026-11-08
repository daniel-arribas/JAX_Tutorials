"""
Immutability.py 

In this tutorial we will work on the Immutability aspect of JAX

We first import the required dependencies and print the JAX version on the screen

"""

import jax
import jax.numpy as jnp
import numpy as np

print(f"📌 JAX Version: {jax.__version__}")
print(f"💻 Executing on Device: {jax.default_backend().upper()}")

"""
----------------------------------------------------------------------------------------------------------------------
Then we try to mutate to see what happens

"""


# Create a standard JAX array filled with zeros
try:
    jax_array = jnp.zeros((3, 3))
    print("Trying to mutate the array in-place like standard NumPy...")
    
    # This line triggers the core design constraint of JAX
    jax_array[0, 0] = 42.0 
    
except TypeError as error:
    print("\n❌ Caught expected JAX TypeError:")
    print(error)
    print("\n💡 Conclusion: JAX arrays are immutable. In-place assignment is prohibited.")

"""
---------------------------------------------------------------------------------------------------------------
We try to do the same the FUNCTIONAL way
"""

# Reset our baseline array
original_array = jnp.zeros((3, 3))

# Apply the JAX functional update mechanism
# This returns an entirely new array reference with the modification applied
modified_array = original_array.at[0, 0].set(42.0)

print("--- 🟢 Original Array Reference (Unchanged) ---")
print(original_array)

print("\n--- 🟢 New Array Reference (With Modification) ---")
print(modified_array)

# Double check that they point to different values
print(f"\nAre both arrays identical? {jnp.array_equal(original_array, modified_array)}")

"""
----------------------------------------------------------------------------------------------------------------------------------
It may look that we are wasting memory by creating a new copy of the array but we can actually use the same name for the array and 
XLA will intellgently assign the same mamory spots economizng resources thanks to the Python Garbage Collector
"""

x = jnp.zeros((3, 3))

print("Array before modification")
print(x)

#Modifying Array

x = x.at[0, 0].set(333)
x = x.at[0, 2].set(555)
print("Array after modification")
print(x)




