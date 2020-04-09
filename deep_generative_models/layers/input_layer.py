from typing import Optional

from torch import Tensor
from torch.nn import Module


class InputLayer(Module):

    def forward(self, inputs: Tensor, condition: Optional[Tensor] = None) -> Tensor:
        raise NotImplementedError

    def get_output_size(self) -> int:
        raise NotImplementedError
