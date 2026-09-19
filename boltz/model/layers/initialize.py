"""Utility functions for initializing weights and biases."""

# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import math

import numpy as np
import torch
from scipy.stats import truncnorm

from boltz.opt import enabled


def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f


def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    torch.nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def bias_init_zero_(bias):
    with torch.no_grad():
        bias.fill_(0.0)


def bias_init_one_(bias):
    with torch.no_grad():
        bias.fill_(1.0)


def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)


#: Initializers this module owns. ``lecun_normal_init_`` and ``he_normal_init_``
#: sample through scipy, which is by far the most expensive of them.
_OWN_INITIALIZERS = (
    "trunc_normal_init_", "lecun_normal_init_", "he_normal_init_",
    "glorot_uniform_init_", "final_init_", "gating_init_", "bias_init_zero_",
    "bias_init_one_", "normal_init_", "ipa_point_weights_init_",
)

#: torch modules whose ``reset_parameters`` writes only registered parameters.
_RESET_PARAMETER_TYPES = (
    torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm,
    torch.nn.Conv1d, torch.nn.Conv2d,
)


@contextlib.contextmanager
def skip_parameter_init():
    """Build modules without initializing weights a checkpoint will overwrite.

    Constructing Boltz-2 samples every weight in the model — including four
    scipy truncated normals — and a ``strict=True`` checkpoint load then writes
    over all of it. Inside this context those writes do not happen, so the
    parameters come out of ``torch.empty`` and the load fills them. The result
    is identical *provided* the load is strict, which is the only place this is
    used: strict loading fails by name on any parameter the checkpoint does not
    supply, so an uninitialized tensor can never reach a forward pass.

    Never use this around a model that will be trained from scratch.
    """
    if not enabled("ctorskip"):
        yield
        return

    module = globals()
    saved = {name: module[name] for name in _OWN_INITIALIZERS}
    saved_resets = {
        cls: cls.reset_parameters for cls in _RESET_PARAMETER_TYPES
    }
    try:
        for name in _OWN_INITIALIZERS:
            module[name] = lambda *args, **kwargs: None
        for cls in _RESET_PARAMETER_TYPES:
            cls.reset_parameters = lambda self: None
        yield
    finally:
        module.update(saved)
        for cls, reset in saved_resets.items():
            cls.reset_parameters = reset
