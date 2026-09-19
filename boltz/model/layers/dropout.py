import torch
from torch import Tensor

from boltz.opt import enabled


class _IdentityMask:
    """Stands in for an all-ones dropout mask, multiplying to the operand itself.

    In eval ``get_dropout_mask`` builds a tensor of exact 1.0s and every call
    site uses it only as ``mask * update``. Returning this sentinel instead
    keeps all seventeen call sites unchanged while skipping both the
    ``[B, N, N, 1]`` allocation and the broadcast multiply over the pair
    representation. Multiplying by exactly 1.0 is the identity in IEEE-754,
    including on signed zeros and infinities, so the residual added downstream
    is bitwise the one stock Boltz-2 adds.
    """

    __slots__ = ()

    def __mul__(self, other: Tensor) -> Tensor:
        return other

    def __rmul__(self, other: Tensor) -> Tensor:
        return other

    def __repr__(self) -> str:
        return "<all-ones dropout mask>"


IDENTITY_MASK = _IdentityMask()


def get_dropout_mask(
    dropout: float,
    z: Tensor,
    training: bool,
    columnwise: bool = False,
) -> Tensor:
    """Get the dropout mask.

    Parameters
    ----------
    dropout : float
        The dropout rate
    z : torch.Tensor
        The tensor to apply dropout to
    training : bool
        Whether the model is in training mode
    columnwise : bool, optional
        Whether to apply dropout columnwise

    Returns
    -------
    torch.Tensor
        The dropout mask

    """
    dropout = dropout * training
    if dropout == 0.0 and enabled("resid"):
        # Inference: the mask below would be all ones divided by one.
        return IDENTITY_MASK
    v = z[:, 0:1, :, 0:1] if columnwise else z[:, :, 0:1, 0:1]
    d = torch.rand(v.shape, dtype=torch.float32, device=v.device) >= dropout
    d = d * 1.0 / (1.0 - dropout)
    return d
