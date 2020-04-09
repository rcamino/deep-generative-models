from typing import Optional

from torch import Tensor
from torch.nn import Module

from deep_generative_models.layers.hidden_layers import HiddenLayersFactory, HiddenLayers
from deep_generative_models.layers.input_layer import InputLayer
from deep_generative_models.layers.output_layer import OutputLayerFactory


class FeedForward(Module):
    input_layer: InputLayer
    hidden_layers: HiddenLayers
    output_layer: Module

    def __init__(self, input_layer: InputLayer, hidden_layers_factory: HiddenLayersFactory,
                 output_layer_factory: OutputLayerFactory, default_hidden_activation: Optional[Module] = None) -> None:
        super(FeedForward, self).__init__()

        self.input_layer = input_layer

        self.hidden_layers = hidden_layers_factory.create(input_layer.get_output_size(),
                                                          default_activation=default_hidden_activation)

        self.output_layer = output_layer_factory.create(self.hidden_layers.get_output_size())

    def forward(self, inputs: Tensor, condition: Optional[Tensor] = None) -> Tensor:
        transformed_inputs = self.input_layer(inputs, condition=condition)
        hidden = self.hidden_layers(transformed_inputs)
        return self.output_layer(hidden)
