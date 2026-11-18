"""Convolutional network with complex parameters for 2D lattices (jVMC_exp / jvmc-2.0).

Same architecture as ``jVMC_exp.nets.CpxCNN``, but frozen into the experiment
folder and specialized to a two-dimensional sample shape ``(Lx, Ly)``:

    log psi(s) = sum_{x,y,c} f( (W * s)_{x,y,c} ) / sqrt(Lx * Ly * C_last)

Convolutions with periodic ("wrap") padding make the ansatz invariant under
lattice translations by construction, so no symmetry projector is needed for
the translation group.

The output is a scalar, complex log-amplitude and the parameters are complex,
i.e. the ansatz is holomorphic.
"""

from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp

import jVMC_exp.global_defs as global_defs
import jVMC_exp.nets.activation_functions as act_funs
from jVMC_exp.nets.initializers import cplx_variance_scaling, init_fn_args


class CpxCNN2D(nn.Module):
    """Complex-parameter CNN on a 2D lattice.

    Initialization arguments:
        * ``F``: Filter diameter (per spatial direction).
        * ``channels``: Number of channels per layer.
        * ``actFun``: Non-linear activation functions. The last entry is reused
          for all remaining layers.
        * ``bias``: Whether to use biases.
        * ``firstLayerBias``: Whether to use a bias in the first layer. A bias
          in the first layer only adds a constant to ``log psi`` and is
          therefore redundant.
        * ``periodicBoundary``: Periodic ("wrap") instead of zero padding.
    """

    Lx: int = 4
    Ly: int = 4
    F: Sequence[int] = (3, 3)
    channels: Sequence[int] = (12,)
    actFun: Sequence[callable] = (act_funs.poly6,)
    bias: bool = True
    firstLayerBias: bool = False
    periodicBoundary: bool = True

    @nn.compact
    def __call__(self, s):
        # A symmetry projector (jVMC_exp.symmetry.ProjectedOrbitNet) hands the
        # configuration to the network in raveled form, hence the reshape.
        s = jnp.reshape(s, (self.Lx, self.Ly))

        # Padding size must be filter diameter - 1. For periodic boundaries the
        # padding is one-sided, which yields a circular convolution.
        pads = [(0, 0)]
        for f in self.F:
            pads.append((0, f - 1) if self.periodicBoundary else (f - 1, f - 1))
        pads.append((0, 0))

        bias = [self.bias] * len(self.channels)
        bias[0] = self.firstLayerBias

        actFun = list(self.actFun)
        actFun += [self.actFun[-1]] * (len(self.channels) - len(actFun))

        init_args = init_fn_args(
            param_dtype=global_defs.DT_PARAMS_CPX,
            kernel_init=cplx_variance_scaling,
        )

        # Add batch and feature dimension: (1, Lx, Ly, 1)
        x = jnp.expand_dims(jnp.expand_dims(2 * s - 1, axis=0), axis=-1)

        for c, f, b in zip(self.channels, actFun, bias):
            x = jnp.pad(x, pads, "wrap" if self.periodicBoundary else "constant")
            x = f(
                nn.Conv(
                    features=c,
                    kernel_size=tuple(self.F),
                    strides=(1,) * len(self.F),
                    padding=[(0, 0)] * len(self.F),
                    use_bias=b,
                    **init_args,
                )(x)
            )

        # Sum over lattice sites (translation invariance) and channels.
        nrm = jnp.sqrt(jnp.prod(jnp.array(x.shape[1:])))

        return jnp.sum(x, dtype=x.dtype) / nrm
