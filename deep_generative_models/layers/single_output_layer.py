from torch import Tensor
from torch.nn import Module, Linear, Sequential

from typing import Optional

from deep_generative_models.layers.output_layer import OutputLayerFactory


class SingleOutputLayer(Module):
    model: Module

    def __init__(self, previous_layer_size: int, output_size: int, activation: Optional[Module] = None) -> None:
        super(SingleOutputLayer, self).__init__()

        if activation is None:
            self.model = Linear(previous_layer_size, output_size)
        else:
            self.model = Sequential(Linear(previous_layer_size, output_size), activation)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.model(inputs)


class SingleOutputLayerFactory(OutputLayerFactory):
    output_size: int
    activation: Optional[Module]

    def __init__(self, output_size: int, activation: Optional[Module] = None) -> None:
        self.output_size = output_size
        self.activation = activation

    def create(self, input_size: int) -> Module:
        return SingleOutputLayer(input_size, self.output_size, self.activation)
