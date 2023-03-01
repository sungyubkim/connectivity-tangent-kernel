# Role of files

* Code Snippets

  * ```cs.py``` : a python implementation of Connectivity Sharpness

  * ```lanczos.py```: a python implementation of Lanczos iteration

  * ```rto.py```: a python implementation of Randomize-Then-Optimize (RTO) method for LL and CL

* Helper scripts
  * ```tool.py```: an auxiliary codes for managing parameters in Haiku
  * ```mp.py```:an auxiliary codes for data parallelization in JAX.



# Requirements

```bash
tensorflow                   2.9.1
tensorflow-datasets          4.6.0
jax                          0.3.17
jaxlib                       0.3.15+cuda11.cudnn82
jaxline                      0.0.5
flax                         0.5.3
dm-haiku                     0.0.9.dev0
```