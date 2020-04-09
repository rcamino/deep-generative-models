from typing import Tuple

from torch import Tensor
from torch.nn import Module


class View(Module):

    shape: Tuple[int, ...]

    def __init__(self, *shape: int) -> None:
        super(View, self).__init__()
        self.shape = shape

    def forward(self, inputs) -> Tensor:
        return inputs.view(*self.shape)
