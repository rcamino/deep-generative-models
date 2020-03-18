from typing import Optional, List

from torch import Tensor
from torch.nn import Module, Sequential, Linear, ReLU

from deep_generative_models.layers.multi_input import MultiInputLayer
from deep_generative_models.metadata import Metadata


class Encoder(Module):

    hidden_layers: Sequential

    def __init__(self, input_size: int, code_size: int, hidden_sizes: List[int] = (),
                 metadata: Optional[Metadata] = None) -> None:
        super(Encoder, self).__init__()

        layers = []

        if metadata is None:
            previous_layer_size = input_size
        else:
            multi_input_layer = MultiInputLayer(metadata)
            layers.append(multi_input_layer)
            previous_layer_size = multi_input_layer.size

        layer_sizes = list(hidden_sizes) + [code_size]
        hidden_activation = ReLU()

        for layer_size in layer_sizes:
            layers.append(Linear(previous_layer_size, layer_size))
            layers.append(hidden_activation)
            previous_layer_size = layer_size

        self.hidden_layers = Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.hidden_layers(inputs)
