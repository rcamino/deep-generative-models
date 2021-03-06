from typing import List, Optional, Any

from torch import Tensor
from torch.nn import Module, Linear, BatchNorm1d, Sequential

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.component_factory import MultiComponentFactory
from deep_generative_models.metadata import Metadata


class HiddenLayers(Module):
    layers: Sequential
    output_size: int

    def __init__(self, layers: Sequential, output_size: int) -> None:
        super(HiddenLayers, self).__init__()
        self.layers = layers
        self.output_size = output_size

    def get_output_size(self) -> int:
        return self.output_size

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class ShortcutHiddenLayer(Module):
    block: Sequential

    def __init__(self, block: Sequential) -> None:
        super(ShortcutHiddenLayer, self).__init__()
        self.block = block

    def forward(self, inputs: Tensor) -> Tensor:
        return self.block(inputs) + inputs


class HiddenLayersFactory:
    sizes: List[int]
    activation: Optional[Module]
    bn_decay: float
    shortcut: bool

    def __init__(self, sizes: List[int] = (), activation: Optional[Module] = None, bn_decay: float = 0,
                 shortcut: bool = False) -> None:
        self.sizes = sizes
        self.activation = activation
        self.bn_decay = bn_decay
        self.shortcut = shortcut

    def create(self, input_size: int, default_activation: Optional[Module] = None) -> HiddenLayers:
        layers = []
        previous_layer_size = input_size

        for layer_number, layer_size in enumerate(self.sizes):
            block_layers = [Linear(previous_layer_size, layer_size)]

            # batch normalization only if defined and if it is not the first layer
            if layer_number > 0 and self.bn_decay > 0:
                block_layers.append(BatchNorm1d(layer_size, momentum=(1 - self.bn_decay)))

            # activation
            if self.activation is not None:
                block_layers.append(self.activation)
            elif default_activation is not None:
                block_layers.append(default_activation)

            # block
            block = Sequential(*block_layers)

            # shortcut
            if self.shortcut:
                layers.append(ShortcutHiddenLayer(block))
            # no-shortcut
            else:
                layers.append(block)

            # move to next layer
            previous_layer_size = layer_size

        # an empty sequential module just works as the identity
        return HiddenLayers(Sequential(*layers), previous_layer_size)


class PartialHiddenLayersFactory(MultiComponentFactory):

    def optional_arguments(self) -> List[str]:
        return ["sizes", "bn_decay", "activation", "shortcut"]

    def create(self, architecture: Architecture, metadata: Metadata, arguments: Configuration) -> Any:
        optional = arguments.get_all_defined(["sizes", "bn_decay", "shortcut"])

        if "activation" in arguments:
            activation_configuration = arguments.activation
            optional["activation"] = self.create_other(activation_configuration.factory, architecture, metadata,
                                                       activation_configuration.get("arguments", {}))

        return HiddenLayersFactory(**optional)
