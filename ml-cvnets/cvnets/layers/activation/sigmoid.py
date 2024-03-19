#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

from torch import Tensor, nn

from cvnets.layers.activation import register_act_fn


@register_act_fn(name="sigmoid")
class Sigmoid(nn.Sigmoid):
    """
    Applies the sigmoid function
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
