"""
In this case we will optimize a Low-Pass RC filter using JAX

"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

""" 
---------------------------------------------------------------------
We define the simulation global parameters

"""
R_init = 500.0 # resistance in ohms
C_init= (100.0e-9) # capacitance in farads

params = (R_init, C_init)

lr_coeff = 0.5

# R works in hundreds, needs a moderate step
lr_R = 10.0*lr_coeff       

# C is in the order of 10^-7, needs a microscopic step
lr_C = 1e-13*lr_coeff



target_freq = 50000
w_target = 2.0 * jnp.pi * target_freq
target_amp = 0.8



""" 
---------------------------------------------------------------------
For a defined RC filter topology we define its Transfer Function

"""

def transfer_function(R, C, w):

    amp = (1/jnp.sqrt(1+(w*R*C)**2))
    phase = -(jnp.arctan(w*R*C))
    
    return amp, phase

""" 
---------------------------------------------------------------------------------------
We define a Loss Function so that the amplitude at the target frequency is exactly 0.8

"""

def loss(params, w_target, target_amp):
    R, C = params
    amp, _ = transfer_function(R, C, w_target)
    
    return ((target_amp-amp)**2)
"""
----------------------------------------------------------------------------------------
Automatic Differentiation with JAX

"""
grad_loss = jax.grad(loss)

"""
----------------------------------------------------------------------------------------
JIT-compiled optimization step: We define a traditional gradient descent

"""

@jax.jit
def train_step(params, lr_R, lr_C, w_target, target_amp):
    grads = grad_loss(params, w_target, target_amp)
    # Descenso de gradiente clásico: params = params - lr * grads
    new_params = (
        jnp.clip(params[0] - lr_R * grads[0], min=1.0), 
        jnp.clip(params[1] - lr_C * grads[1], min=1e-12)
        ) # we jnp.clip the values to avoid negative values
    return new_params

""" 
-----------------------------------------------------------------------------------------
Optimization Loop: We define it using a for loop to avoid using lax.scan by now

"""

for step in range(100000):
    params = train_step(params, lr_R, lr_C, w_target, target_amp)
    if step % 5 == 0:
        current_loss = loss(params, w_target, target_amp)
        print(f"Step {step:3d} | Loss: {current_loss:.6e} | R: {params[0]:.2f} Ω | C: {params[1]*1e9:.2f} nF")

R_opt, C_opt = params










