from typing import Any, Callable
from functools import partial
import jax
import jax.numpy as jnp
from flax import struct
import optax

from absl import flags
from jax.flatten_util import ravel_pytree

from absl import flags
FLAGS = flags.FLAGS

class Trainer(struct.PyTreeNode):
    step: int
    apply_fn: Callable = struct.field(pytree_node=False)
    tx: Callable = struct.field(pytree_node=False)
    params: Any = None
    state: Any = None
    opt_state: Any = None
    
    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        opt_state = tx.init(params)
        return cls(step=0, apply_fn=apply_fn, params=params, tx=tx, opt_state=opt_state, **kwargs)
    
    def apply_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(step=self.step+1, params=new_params, opt_state=new_opt_state, **kwargs)
    
def select_tree(pred: jnp.ndarray, a, b):
    assert pred.ndim == 0 and pred.dtype == jnp.bool_, "expected boolean scalar"
    return jax.tree_map(partial(jax.lax.select, pred), a, b)
    
class TrainerPert(Trainer):
    offset : Any = None

def params_to_vec(param, unravel=False):
    vec_param, unravel_fn = ravel_pytree(param)
    if unravel:
        return vec_param, unravel_fn
    else:
        return vec_param
    
def forward(params, trainer, input_, rng=None, train=True):
    res, _ = trainer.apply_fn(params, trainer.state, rng, input_, train)
    return res

def init_trainer_ft_lin(trainer, init_params=None):
    vec_params, unravel_fn = params_to_vec(trainer.params, True)
    if init_params is None:
        init_params = jnp.zeros_like(vec_params)
    tx = optax.chain(optax.adam(learning_rate=FLAGS.ft_lr))
    trainer_ft = TrainerPert.create(
        apply_fn=trainer.apply_fn,
        state=trainer.state,
        offset=trainer.params,
        params=unravel_fn(init_params),
        tx=tx,
    )
    return trainer_ft