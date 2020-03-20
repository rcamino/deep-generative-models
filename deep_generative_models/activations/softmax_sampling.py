from torch import Tensor
from torch.nn import Module

from torch.nn.functional import softmax

from torch.distributions.one_hot_categorical import OneHotCategorical


class SoftmaxSampling(Module):

    def forward(self, inputs: Tensor) -> Tensor:
        if self.training:
            return softmax(inputs, dim=1)
        else:
            return OneHotCategorical(logits=inputs).sample()
