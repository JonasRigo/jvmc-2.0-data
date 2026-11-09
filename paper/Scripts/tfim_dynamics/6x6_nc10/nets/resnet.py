"""Convolutional ResNet with real parameters and complex output (jVMC_exp / jvmc-2.0).

Ported from ``old_scripts/0encap_folder/crit_tfim_update/perturbed_start_delta=1e_4_full/nets/CNN_resnet.py``
(and its 2D variant in ``old_scripts/crit_tfim_2D/nets/Resnet.py``) to the
jvmc-2.0 API. Changes w.r.t. the original:

* ``jVMC`` -> ``jVMC_exp``; ``global_defs.tReal`` -> ``global_defs.DT_PARAMS_REAL``,
  ``jnp.complex128`` -> ``global_defs.DT_OUT_CPX``.
* Removed the unused ``jVMC.util.symmetries.LatticeSymmetry`` import (the module
  no longer exists; symmetries live in ``jVMC_exp.symmetry``).
* ``padding="CIRCULAR"`` for periodic boundary conditions instead of ``"SAME"``
  (zero padding), selectable via ``periodicBoundary``.

The last block's channels are split into real and imaginary part, so
``channels[-1]`` must be even. Because the output is a ``logsumexp`` over sites
and channels, the ansatz is translation invariant by construction (for periodic
padding).

Real parameters with complex output means the ansatz is *not* holomorphic, so
the TDVP has to be set up with ``make_real=True``.
"""

from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

import jVMC_exp.global_defs as global_defs


class ResNet(nn.Module):
    """Convolutional neural network with residual connections.

    Initialization arguments:
        * ``F``: Filter diameter (per spatial direction).
        * ``channels``: Number of channels per residual block. Two convolutions
          are applied per block; ``channels[-1]`` must be even.
        * ``bias``: Whether to use biases.
        * ``periodicBoundary``: Circular instead of zero padding.
    """

    Lx: int = 4
    Ly: int = 4
    F: Sequence[int] = (3, 3)
    channels: Sequence[int] = (16,)
    strides: Sequence[int] = (1, 1)
    bias: bool = True
    periodicBoundary: bool = True

    @nn.compact
    def __call__(self, s):
        padding = "CIRCULAR" if self.periodicBoundary else "SAME"
        nsites = s.size

        # A symmetry projector (jVMC_exp.symmetry.ProjectedOrbitNet) hands the
        # configuration to the network in raveled form, hence the reshape.
        s = jnp.reshape(s, (self.Lx, self.Ly))

        # Add batch and feature dimension: (1, Lx, Ly, 1)
        x = jnp.expand_dims(jnp.expand_dims(2 * s - 1, axis=0), axis=-1)

        for nblock in range(len(self.channels)):

            residual = x
            x = x / np.sqrt(nblock + 1, dtype=global_defs.DT_PARAMS_REAL)

            if nblock == 0:
                x = x / np.sqrt(2, dtype=global_defs.DT_PARAMS_REAL)
            else:
                x = nn.gelu(x)

            x = nn.Conv(
                features=self.channels[nblock],
                kernel_size=tuple(self.F),
                padding=padding,
                strides=self.strides,
                use_bias=self.bias,
                param_dtype=global_defs.DT_PARAMS_REAL,
                dtype=global_defs.DT_PARAMS_REAL,
                kernel_init=jax.nn.initializers.he_normal(),
                bias_init=jax.nn.initializers.zeros,
            )(x)

            x = nn.gelu(x)

            x = nn.Conv(
                features=self.channels[nblock],
                kernel_size=tuple(self.F),
                padding=padding,
                strides=self.strides,
                # The bias of the last convolution only shifts log psi by a
                # constant, hence it is dropped.
                use_bias=self.bias and not (nblock == len(self.channels) - 1),
                param_dtype=global_defs.DT_PARAMS_REAL,
                dtype=global_defs.DT_PARAMS_REAL,
                kernel_init=jax.nn.initializers.he_normal(),
                bias_init=jax.nn.initializers.zeros,
            )(x)

            # Broadcasts the single input channel onto all channels of the
            # first block.
            x = x + residual

        x = x / np.sqrt(len(self.channels), dtype=global_defs.DT_PARAMS_REAL)

        # Interpret one half of the channels as real, the other as imaginary part.
        x = jax.lax.complex(x[..., :(x.shape[-1] // 2)], x[..., (x.shape[-1] // 2):])
        x = x.astype(global_defs.DT_OUT_CPX)

        # Sum over sites and channels in log space (translation invariance).
        return jax.scipy.special.logsumexp(x) - jnp.log(self.channels[-1] * nsites)
