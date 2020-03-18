from torch import Tensor
from torch.nn import Module


class OutputLayer(Module):
    """
    This is just a simple abstract class for single and multi output layers.
    Both need to have the same interface.
    """

    def forward(self, inputs: Tensor, training: bool = None):
        raise NotImplementedError
