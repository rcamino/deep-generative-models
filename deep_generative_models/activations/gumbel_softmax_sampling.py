from torch import Tensor
from torch.nn import Module

from torch.nn.functional import gumbel_softmax


class GumbelSoftmaxSampling(Module):
    temperature: float

    def __init__(self, temperature: float) -> None:
        super(GumbelSoftmaxSampling, self).__init__()
        self.temperature = temperature

    def forward(self, inputs: Tensor) -> Tensor:
        return gumbel_softmax(inputs, hard=not self.training, tau=self.temperature)
