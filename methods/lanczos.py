import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')
from absl import flags
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

import tool, mp

flags.DEFINE_float('scale_factor', 1.0, help='scale factor')
flags.DEFINE_bool('use_connect', False, help='use connectivity or parameter')
flags.DEFINE_integer('sharpness_rand_proj_dim', 10, help='random projection dimension for sharpness estimation')
FLAGS = flags.FLAGS

@jax.pmap
def mvp_batch(v, trainer, batch):
    # apply JVP -> VJP to compute J^T J v (Matrix-vector multiplication)
    vec_params, unravel_fn = tool.params_to_vec(trainer.params, True)
    
    if FLAGS.use_connect:
        multiplier = vec_params
    else:
        multiplier = jnp.ones_like(vec_params)
        
    f = lambda x: tool.forward(x, trainer, batch['x'], train=True)
    _, res = jax.jvp(f, [unravel_fn(vec_params)], [unravel_fn(v * multiplier)])
    _, f_vjp = jax.vjp(f, unravel_fn(vec_params))
    res = f_vjp(res)
    res = tool.params_to_vec(res) * multiplier
    return res

def mvp(v, trainer, ds_train):
    res = 0.
    for batch in ds_train:
        res += mvp_batch(mp.replicate(v), trainer, batch).sum(axis=0)
    return res

def lanczos(trainer, ds_train, rng):
    # Modified lanczos alogrithm of https://github.com/google/spectral-density/blob/master/jax/lanczos.py for recent jax ver.
    rand_proj_dim = FLAGS.sharpness_rand_proj_dim
    vec_params, unravel_fn = tool.params_to_vec(mp.unreplicate(trainer).params, True)
    
    tridiag = jnp.zeros((rand_proj_dim, rand_proj_dim))
    vecs = jnp.zeros((rand_proj_dim, len(vec_params)))
    
    init_vec = jax.random.normal(rng, shape=vec_params.shape)
    init_vec = init_vec / jnp.linalg.norm(init_vec)
    vecs = vecs.at[0].set(init_vec)
    
    beta = 0
    for i in tqdm(range(rand_proj_dim)):
        v = vecs[i, :]
        if i == 0:
            v_old = 0
        else:
            v_old = vecs[i -1, :]
        
        w = mvp(v, trainer, ds_train)
        w = w - beta * v_old
        
        alpha = jnp.dot(w, v)
        tridiag = tridiag.at[i, i].set(alpha)
        w = w - alpha * v
        
        for j in range(i):
            tau = vecs[j, :]
            coef = np.dot(w, tau)
            w += - coef * tau
            
        beta = jnp.linalg.norm(w)
        
        if (i + 1) < rand_proj_dim:
            tridiag = tridiag.at[i, i+1].set(beta)
            tridiag = tridiag.at[i+1, i].set(beta)
            vecs = vecs.at[i+1].set(w/beta)
            
    return tridiag, vecs