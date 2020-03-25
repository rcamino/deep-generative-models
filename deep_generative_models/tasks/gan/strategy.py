from torch import Tensor

from deep_generative_models.architecture import Architecture


class GANStrategy:
    architecture: Architecture

    def __init__(self, architecture: Architecture):
        self.architecture = architecture

    def wrap_real_features(self, real_features: Tensor) -> Tensor:
        raise NotImplementedError

    def wrap_generator_outputs(self, generator_outputs: Tensor) -> Tensor:
        raise NotImplementedError


class DiscriminateCodes(GANStrategy):

    def wrap_real_features(self, real_features: Tensor) -> Tensor:
        return self.architecture.autoencoder.encode(real_features)

    def wrap_generator_outputs(self, generator_outputs: Tensor) -> Tensor:
        return generator_outputs


class DecodeGeneratorOutputs(GANStrategy):

    def wrap_real_features(self, real_features: Tensor) -> Tensor:
        return real_features

    def wrap_generator_outputs(self, generator_outputs: Tensor) -> Tensor:
        return self.architecture.autoencoder.decode(generator_outputs)


class VanillaGAN(GANStrategy):

    def wrap_real_features(self, real_features: Tensor) -> Tensor:
        return real_features

    def wrap_generator_outputs(self, generator_outputs: Tensor) -> Tensor:
        return generator_outputs


strategy_class_by_name = {
    "DecodeGeneratorOutputs": DecodeGeneratorOutputs,
    "DiscriminateCodes": DiscriminateCodes,
    "VanillaGAN": VanillaGAN,
}


def create_gan_strategy(architecture: Architecture, name: str) -> GANStrategy:
    return strategy_class_by_name[name](architecture)
