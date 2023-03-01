from functools import partial
import jax
import jax.numpy as jnp
import tool

# evaluate connectivity sharpness

def eval_ctk_trace(forward, params, state, batch, rng, max_iter, num_classes):
  vec_params = tool.params_to_vec(params)
  f = partial(forward, state=state, input_=batch['x'])
  _, f_vjp = jax.vjp(f, params)
  
  def body_fn(_, a):
    res, rng = a
    _, rng = jax.random.split( rng )
    v = jax.random.rademacher(rng, (batch['x'].shape[0], num_classes), jnp.float32)
    j_p = tool.params_to_vec(f_vjp(v)) * vec_params
    tr_ctk = jnp.sum(jnp.square(j_p)) / batch['x'].shape[0]
    res += tr_ctk / max_iter
    return (res, rng)
  
  a = jax.lax.fori_loop(0, max_iter, body_fn, (0., rng))
  res, rng = a
  return res