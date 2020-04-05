from typing import Any, Dict, List

from torch import Tensor, empty_like
from torch.nn import Module

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.metadata import Metadata
from deep_generative_models.component_factory import MultiComponentFactory, ComponentFactory
from deep_generative_models.models.autoencoder import AutoEncoder


class DeNoisingAutoencoder(Module):

    autoencoder: AutoEncoder
    noise_mean: float
    noise_std: float

    def __init__(self, autoencoder: AutoEncoder, noise_mean: float = 0, noise_std: float = 1) -> None:
        super(DeNoisingAutoencoder, self).__init__()

        self.autoencoder = autoencoder
        self.noise_mean = noise_mean
        self.noise_std = noise_std

    def forward(self, inputs: Tensor) -> Dict[str, Tensor]:
        outputs = self.encode(inputs)
        outputs["reconstructed"] = self.decode(outputs["code"])
        return outputs

    def encode(self, inputs: Tensor) -> Dict[str, Tensor]:
        noisy = self.add_noise(inputs)
        outputs = self.autoencoder.encode(noisy)
        outputs["noisy"] = noisy
        return outputs

    def add_noise(self, inputs: Tensor) -> Tensor:
        return empty_like(inputs).normal_(self.noise_mean, self.noise_std) + inputs

    def decode(self, code: Tensor) -> Tensor:
        return self.autoencoder.decode(code)


class DeNoisingAutoencoderFactory(MultiComponentFactory):
    factory_name: str

    def __init__(self, factory_by_name: Dict[str, ComponentFactory], factory_name: str):
        super(DeNoisingAutoencoderFactory, self).__init__(factory_by_name)
        self.factory_name = factory_name

    def optional_arguments(self) -> List[str]:
        return ["noise_mean", "noise_std"]

    def create(self, architecture: Architecture, metadata: Metadata, arguments: Configuration) -> Any:
        autoencoder = self.create_other(self.factory_name, metadata, arguments, )
        return DeNoisingAutoencoder(autoencoder, **arguments.get_all_defined(self.optional_arguments()))
