from typing import Any, Dict, List

from torch import Tensor
from torch.nn import Module

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.metadata import Metadata
from deep_generative_models.component_factory import MultiComponentFactory, ComponentFactory
from deep_generative_models.models.autoencoder import AutoEncoder


class DeNoisingAutoencoder(Module):
    noise_layer: Module
    autoencoder: AutoEncoder

    def __init__(self, noise_layer: Module, autoencoder: AutoEncoder) -> None:
        super(DeNoisingAutoencoder, self).__init__()

        self.noise_layer = noise_layer
        self.autoencoder = autoencoder

    def forward(self, inputs: Tensor, **additional_inputs: Tensor) -> Dict[str, Tensor]:
        outputs = self.encode(inputs, **additional_inputs)
        outputs["reconstructed"] = self.decode(outputs["code"], **additional_inputs)
        return outputs

    def encode(self, inputs: Tensor, **additional_inputs: Tensor) -> Dict[str, Tensor]:
        noisy = self.noise_layer(inputs)
        outputs = self.autoencoder.encode(noisy, **additional_inputs)
        outputs["noisy"] = noisy
        return outputs

    def decode(self, code: Tensor, **additional_inputs: Tensor) -> Tensor:
        return self.autoencoder.decode(code, **additional_inputs)


class DeNoisingAutoencoderFactory(MultiComponentFactory):
    factory_name: str

    def __init__(self, factory_by_name: Dict[str, ComponentFactory], factory_name: str):
        super(DeNoisingAutoencoderFactory, self).__init__(factory_by_name)
        self.factory_name = factory_name

    def mandatory_arguments(self) -> List[str]:
        return ["encoder", "decoder"]

    def optional_arguments(self) -> List[str]:
        return ["noise_layer"]

    def create(self, architecture: Architecture, metadata: Metadata, arguments: Configuration) -> Any:
        # separate arguments
        noise_layer_arguments = None
        autoencoder_arguments = Configuration()
        for argument_name, argument_value in arguments.items():
            if argument_name == "noise_layer":
                noise_layer_arguments = argument_value
            else:
                autoencoder_arguments[argument_name] = argument_value

        noise_layer = self.create_other(noise_layer_arguments.factory,
                                        architecture,
                                        metadata,
                                        noise_layer_arguments.get("arguments", {}))

        autoencoder = self.create_other(self.factory_name, architecture, metadata, autoencoder_arguments)

        return DeNoisingAutoencoder(noise_layer, autoencoder)
