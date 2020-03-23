from torch.nn import Module, ReLU, Sigmoid, Tanh, LeakyReLU, BCELoss, CrossEntropyLoss, MSELoss

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
from deep_generative_models.models.generator import SingleOutputGeneratorFactory, MultiOutputGeneratorFactory
from deep_generative_models.models.initialization import initialize_module


class Architecture(Dictionary[Module]):

    def to_gpu_if_available(self) -> None:
        for name, module in self.items():
            self[name] = to_gpu_if_available(module)

    def to_cpu_if_was_in_gpu(self) -> None:
        for name, module in self.items():
            self[name] = to_cpu_if_was_in_gpu(module)

    def initialize(self):
        for module in self.values():
            initialize_module(module)


factory_by_name = {
    # my activations
    "GumbelSoftmaxSampling": GumbelSoftmaxSamplingFactory(),
    "SoftmaxSampling": ClassFactoryWrapper(SoftmaxSampling),

    # PyTorch activations (could add more)
    "ReLU": ClassFactoryWrapper(ReLU),
    "Sigmoid": ClassFactoryWrapper(Sigmoid),
    "Tanh": ClassFactoryWrapper(Tanh),
    "LeakyReLU": ClassFactoryWrapper(LeakyReLU),

    # PyTorch losses (could add more)
    "BCE": ClassFactoryWrapper(BCELoss),
    "CrossEntropy": ClassFactoryWrapper(CrossEntropyLoss),
    "MSE": ClassFactoryWrapper(MSELoss),
}

factory_by_name["SingleVariableAutoEncoder"] = SingleVariableAutoEncoderFactory(factory_by_name)
factory_by_name["MultiVariableAutoEncoder"] = MultiVariableAutoEncoderFactory(factory_by_name)
factory_by_name["SingleInputEncoder"] = SingleInputEncoderFactory(factory_by_name)
factory_by_name["MultiInputEncoder"] = MultiInputEncoderFactory(factory_by_name)
factory_by_name["SingleOutputDecoder"] = SingleOutputDecoderFactory(factory_by_name)
factory_by_name["MultiOutputDecoder"] = MultiOutputDecoderFactory(factory_by_name)
factory_by_name["SingleInputLayer"] = SingleInputLayerFactory(factory_by_name)
factory_by_name["MultiInputLayer"] = MultiInputLayerFactory(factory_by_name)
factory_by_name["SingleOutputLayer"] = PartialSingleOutputLayerFactory(factory_by_name)
factory_by_name["MultiOutputLayer"] = PartialMultiOutputLayerFactory(factory_by_name)
factory_by_name["SingleOutputGenerator"] = SingleOutputGeneratorFactory(factory_by_name)
factory_by_name["MultiOutputGenerator"] = MultiOutputGeneratorFactory(factory_by_name)
factory_by_name["Discriminator"] = DiscriminatorFactory(factory_by_name)


def create_architecture(metadata: Metadata, configuration: Configuration) -> Architecture:
    architecture = Architecture()
    for name, child_configuration in configuration.architecture.items():
        factory = factory_by_name[child_configuration.factory]
        architecture[name] = factory.create(metadata, configuration, child_configuration.get("arguments", {}))
    return architecture
