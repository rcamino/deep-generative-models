from torch import Tensor
from torch.nn import Module


class ImputationLayer(Module):

    def forward(self, inputs: Tensor, missing_mask: Tensor) -> Tensor:
        raise NotImplementedError
