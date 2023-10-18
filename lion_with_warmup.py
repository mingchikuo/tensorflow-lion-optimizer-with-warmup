"""
Copyright © 2023 Ming-Chi Kuo (Mitchel) and https://github.com/mingchikuo

1. Hi everyone, I'm Ming-Chi Kuo (Mitchel), an AI and software algorithm developer. You can find my work on GitHub at https://github.com/mingchikuo.

2. If you wish to utilize any of the open-source algorithms I have provided for personal or research purposes, 

kindly acknowledge the authorship by crediting me (Ming-Chi Kuo) and including my GitHub profile URL(https://github.com/mingchikuo).

3. Please note that any commercial use of these open-source algorithms is strictly prohibited without my explicit consent.

4. For inquiries or potential collaborations, please feel free to reach out to me via my GitHub profile.

5. Modified by Ming-Chi Kuo(Mitchel) based on 2023 Google Research and https://github.com/google/automl/blob/master/lion/lion_tf2.py. 
This modification does not represent the 2023 Google Research but reflects my own implement.
"""

"""
The optimizer(lion) has a num_warmup_batches hyperparameter that you can set when 
initializing the optimizer. This parameter represents the number of batches 
used for warm-up.

Inside the _prepare_local method, the code calculates a warmup_coeff by 
dividing the current iteration count (self.iterations) by the 
num_warmup_batches. This coefficient starts at 0 and gradually increases 
towards 1 as training progresses.

The learning rate (lr) is then scaled by this warmup_coeff. This means 
that during the warm-up phase, the learning rate is gradually increased 
from its initial value to the full learning rate specified by the user.
"""

import tensorflow as tf

class Lion(tf.keras.optimizers.legacy.Optimizer):
  r"""Optimizer that implements the Lion algorithm."""

  def __init__(self,
               learning_rate=0.0001,
               beta_1=0.9,
               beta_2=0.99,
               wd=0,
               name='lion',
               num_warmup_batches=2000,
               **kwargs):
    """Construct a new Lion optimizer."""

    super(Lion, self).__init__(name, **kwargs)
    self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
    self._set_hyper('beta_1', beta_1)
    self._set_hyper('beta_2', beta_2)
    self._set_hyper('wd', wd)
    # Add num_warmup_epochs hyperparameter
    self._set_hyper('num_warmup_batches', num_warmup_batches)

  def _create_slots(self, var_list):
    # Create slots for the first and second moments.
    # Separate for-loops to respect the ordering of slot variables from v1.
    for var in var_list:
      self.add_slot(var, 'm')

  def _prepare_local(self, var_device, var_dtype, apply_state):

    super(Lion, self)._prepare_local(var_device, var_dtype, apply_state)

    # Get hyperparameters
    beta_1_t = tf.identity(self._get_hyper('beta_1', var_dtype))
    beta_2_t = tf.identity(self._get_hyper('beta_2', var_dtype))
    wd_t = tf.identity(self._get_hyper('wd', var_dtype))

    # Get learning rate
    lr = apply_state[(var_device, var_dtype)]['lr_t']

    # Calculate warmup coefficient  
    num_warmup_batches = self._get_hyper('num_warmup_batches', var_dtype)
    warmup_coeff = tf.cast(self.iterations, tf.float32) / num_warmup_batches
    warmup_coeff = tf.minimum(warmup_coeff, 1.0)

    # Scale learning rate by warmup coefficient
    lr = lr * warmup_coeff

    # Update state with modified lr
    apply_state[(var_device, var_dtype)].update(
        lr=lr,
        beta_1_t=beta_1_t,
        one_minus_beta_1_t=1 - beta_1_t,
        beta_2_t=beta_2_t,
        one_minus_beta_2_t=1 - beta_2_t,
        wd_t=wd_t)

  @tf.function(jit_compile=True)
  def _resource_apply_dense(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
                    self._fallback_apply_state(var_device, var_dtype))

    m = self.get_slot(var, 'm')
    var_t = var.assign_sub(
        coefficients['lr_t'] *
        (tf.math.sign(m * coefficients['beta_1_t'] +
                      grad * coefficients['one_minus_beta_1_t']) +
         var * coefficients['wd_t']))
    with tf.control_dependencies([var_t]):
      m.assign(m * coefficients['beta_2_t'] +
               grad * coefficients['one_minus_beta_2_t'])

  @tf.function(jit_compile=True)
  def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = ((apply_state or {}).get((var_device, var_dtype)) or
                    self._fallback_apply_state(var_device, var_dtype))

    m = self.get_slot(var, 'm')
    m_t = m.assign(m * coefficients['beta_1_t'])
    m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
    m_t = m_t.scatter_add(tf.IndexedSlices(m_scaled_g_values, indices))
    var_t = var.assign_sub(coefficients['lr'] *
                           (tf.math.sign(m_t) + var * coefficients['wd_t']))

    with tf.control_dependencies([var_t]):
      m_t = m_t.scatter_add(tf.IndexedSlices(-m_scaled_g_values, indices))
      m_t = m_t.assign(m_t * coefficients['beta_2_t'] /
                       coefficients['beta_1_t'])
      m_scaled_g_values = grad * coefficients['one_minus_beta_2_t']
      m_t.scatter_add(tf.IndexedSlices(m_scaled_g_values, indices))

  def get_config(self):
    config = super(Lion, self).get_config()
    config.update({
        'learning_rate': self._serialize_hyperparameter('learning_rate'),
        'beta_1': self._serialize_hyperparameter('beta_1'),
        'beta_2': self._serialize_hyperparameter('beta_2'),
        'wd': self._serialize_hyperparameter('wd'),
        'num_warmup_batches': self._serialize_hyperparameter('num_warmup_batches'),
    })
    return config
