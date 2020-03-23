from torch.nn import Module, ReLU, Sigmoid, Tanh, LeakyReLU

from deep_generative_models.activations.gumbel_softmax_sampling import GumbelSoftmaxSamplingFactory
from deep_generative_models.activations.softmax_sampling import SoftmaxSampling
from deep_generative_models.configuration import Configuration
from deep_generative_models.dictionary import Dictionary
from deep_generative_models.factory import ClassFactoryWrapper
from deep_generative_models.gpu import to_gpu_if_available, to_cpu_if_was_in_gpu

from deep_generative_models.layers.multi_input_layer import MultiInputLayerFactory
from deep_generative_models.layers.multi_output_layer import PartialMultiOutputLayerFactory
from deep_generative_models.layers.single_input_layer import SingleInputLayerFactory
from deep_generative_models.layers.single_output_layer import PartialSingleOutputLayerFactory

from deep_generative_models.metadata import Metadata

from deep_generative_models.models.autoencoder import SingleVariableAutoEncoderFactory
from deep_generative_models.models.autoencoder import MultiVariableAutoEncoderFactory
from deep_generative_models.models.decoder import SingleOutputDecoderFactory
from deep_generative_models.models.decoder import MultiOutputDecoderFactory
from deep_generative_models.models.discriminator import DiscriminatorFactory
from deep_generative_models.models.encoder import SingleInputEncoderFactory, MultiInputEncoderFactory
from deep_generative_models.models.generator import SingleVariableGeneratorFactory, MultiVariableGeneratorFactory


class Architecture(Dictionary[Module]):

    def to_gpu_if_available(self) -> None:
        for name, module in self.items():
            self[name] = to_gpu_if_available(module)

    def to_cpu_if_was_in_gpu(self) -> None:
        for name, module in self.items():
            self[name] = to_cpu_if_was_in_gpu(module)


factory_by_name = {
    # my activations
    "gumbel-softmax sampling": GumbelSoftmaxSamplingFactory(),
    "softmax sampling": ClassFactoryWrapper(SoftmaxSampling),

    # PyTorch activations (could add more)
    "ReLU": ClassFactoryWrapper(ReLU),
    "Sigmoid": ClassFactoryWrapper(Sigmoid),
    "Tanh": ClassFactoryWrapper(Tanh),
    "LeakyReLU": ClassFactoryWrapper(LeakyReLU)
}

factory_by_name["single-variable autoencoder"] = SingleVariableAutoEncoderFactory(factory_by_name)
factory_by_name["multi-variable autoencoder"] = MultiVariableAutoEncoderFactory(factory_by_name)
factory_by_name["single-input encoder"] = SingleInputEncoderFactory(factory_by_name)
factory_by_name["multi-input encoder"] = MultiInputEncoderFactory(factory_by_name)
factory_by_name["single-output decoder"] = SingleOutputDecoderFactory(factory_by_name)
factory_by_name["multi-output decoder"] = MultiOutputDecoderFactory(factory_by_name)
factory_by_name["single-input layer"] = SingleInputLayerFactory(factory_by_name)
factory_by_name["multi-input layer"] = MultiInputLayerFactory(factory_by_name)
factory_by_name["single-output layer"] = PartialSingleOutputLayerFactory(factory_by_name)
factory_by_name["multi-output layer"] = PartialMultiOutputLayerFactory(factory_by_name)
factory_by_name["single-output generator"] = SingleVariableGeneratorFactory(factory_by_name)
factory_by_name["multi-output generator"] = MultiVariableGeneratorFactory(factory_by_name)
factory_by_name["discriminator"] = DiscriminatorFactory(factory_by_name)


def create_architecture(metadata: Metadata, configuration: Configuration) -> Architecture:
    architecture = Architecture()
    for name, child_configuration in configuration.architecture.items():
        factory = factory_by_name[child_configuration.factory]
        architecture[name] = factory.create(metadata, configuration, child_configuration.get("arguments", {}))
    return architecture
