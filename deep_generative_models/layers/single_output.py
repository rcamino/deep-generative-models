from torch import Tensor
from torch.nn import Module, Linear, Sequential

from deep_generative_models.layers.output_layer import OutputLayer


class SingleOutputLayer(OutputLayer):
    model: Module

    def __init__(self, previous_layer_size: int, output_size: int, activation: Module = None) -> None:
        super(SingleOutputLayer, self).__init__()

        if activation is None:
            self.model = Linear(previous_layer_size, output_size)
        else:
            self.model = Sequential(Linear(previous_layer_size, output_size), activation)

    def forward(self, inputs: Tensor, training: bool = None) -> Tensor:
        return self.model(inputs)
