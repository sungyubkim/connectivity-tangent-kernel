import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from typing import Any, OrderedDict
from tqdm import tqdm
from absl import flags
from time import time

import tool, mp

FLAGS = flags.FLAGS

def loss_fn(params, num_train, trainer, batch, rng, param_noise, batch_noise):
    vec_params = tool.params_to_vec(params)
    if FLAGS.use_connect:
        new_params = jax.tree_util.tree_map(
            lambda x, y: x * y, params, trainer.offset)
    else:
        new_params = params
        
    f = lambda x: tool.forward(x, trainer, batch['x'], train=True)
    logit, jvp = jax.jvp(f, [trainer.offset], [new_params])
        
    acc = (jnp.argmax(logit + jvp, axis=-1) == jnp.argmax(batch['y'], axis=-1)).astype(int)
    
    loss = 0.5 * ((jvp - batch_noise * batch['y'])**2).sum(axis=-1)
    wd = 0.5 * ((vec_params - param_noise)**2).sum()
    loss_ = loss.mean() + wd / num_train * (FLAGS.sigma/FLAGS.alpha)**2
    param_norm = (vec_params**2).sum()
    return loss_, (loss, acc, wd, param_norm)

@partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(3,4))
def new_opt_step(trainer, batch, rng, sync_grad, num_train, param_noise, data_noise):
    vec_params, unravel_fn = tool.params_to_vec(trainer.params, True)
    batch_noise = data_noise[batch['idx']]
    grad_fn = jax.grad(loss_fn, has_aux=True)
    # compute grad
    grad, (loss_b, acc_b, wd_b, param_norm) = grad_fn(
        trainer.params,
        num_train,
        trainer,
        batch,
        rng,
        param_noise,
        batch_noise,
        )
    grad = tool.params_to_vec(grad)
    grad_norm = jnp.sqrt((grad**2).sum())

    log = [
        ('loss_sgd', loss_b),
        ('acc_sgd', acc_b),
        ('wd_sgd', wd_b),
        ('grad_sgd', grad_norm),
        ('param_sgd', param_norm),
    ]
    log = OrderedDict(log)
    
    # update NN
    if sync_grad:
        grad = jax.lax.pmean(grad, axis_name='batch')
    trainer = trainer.apply_gradients(grads=unravel_fn(grad))
    return log, trainer

def get_posterior_rto(trainer, opt_step, dataset_opt, rng, *args, **kwargs):
    num_devices = jax.device_count()
    num_train = kwargs['num_train']
    num_class = kwargs['num_class']
    sync_grad = not(FLAGS.ft_local)
    vec_params= tool.params_to_vec(mp.unreplicate(trainer).params)
    init_p = jax.pmap(tool.init_trainer_ft_lin)
    
    if FLAGS.ft_local:
        num_stage = int(np.ceil(float(FLAGS.ens_num)/num_devices))
    else:
        num_stage = FLAGS.ens_num
    
    print(f'Start {FLAGS.ens_num} ensemble training')
    posterior = []
    for i in range(num_stage):
        # randomize
        rng, rng_ = jax.random.split(rng)
        if sync_grad:
            param_noise = mp.replicate(jax.random.normal(rng_, vec_params.shape)) * FLAGS.alpha
            rng, rng_ = jax.random.split(rng)
            data_noise = mp.replicate(jax.random.normal(rng_, (num_train, num_class))) * FLAGS.sigma
        else:
            param_noise = jax.random.normal(rng_, (num_devices, *vec_params.shape)) * FLAGS.alpha
            rng, rng_ = jax.random.split(rng)
            data_noise = jax.random.normal(rng_, (num_devices, num_train, num_class)) * FLAGS.sigma
            
        trainer_ft = init_p(trainer, param_noise)
        
        # optimize
        pbar = tqdm(range(FLAGS.ft_step))
        for step in pbar:
            if i==0 and step==1:
                # remove first iteration to exclude compile time
                start_time = time()
            batch_tr = next(dataset_opt)
            rng, rng_ = jax.random.split(rng)
            log, trainer_ft = new_opt_step(
                trainer_ft, 
                batch_tr, 
                jax.random.split(rng_, num_devices), 
                sync_grad, 
                num_train,
                param_noise, 
                data_noise,
                )
            
            log = OrderedDict([(k,f'{np.mean(v):.2f}') for k,v in log.items()])
            log.update({'stage': i})
            log.move_to_end('stage', last=False)
            pbar.set_postfix(log)
            
        end_time = time()
        print(f'Cache time except compile : {end_time - start_time:.4} s')

        print(f'Post-processing members')
        if FLAGS.ft_local:
            # add multiple local models
            for ens_mem in tqdm(range(num_devices)):
                member = mp.unreplicate(trainer_ft, ens_mem)
                if FLAGS.use_connect:
                    member = member.replace(params=jax.tree_util.tree_map(lambda x, y : x * y, member.params, member.offset))
                posterior.append(member)
        else:
            # add single global model
            member = mp.unreplicate(trainer_ft)
            if FLAGS.use_connect:
                member = member.replace(params=jax.tree_util.tree_map(lambda x, y : x * y, member.params, member.offset))
            posterior.append(member)
    return posterior