from torch.nn import LeakyReLU, ReLU, Sigmoid, Softmax, Tanh, BCELoss, CrossEntropyLoss, MSELoss
from torch.optim import Adam, SGD

from deep_generative_models.activations.gumbel_softmax_sampling import GumbelSoftmaxSamplingFactory
from deep_generative_models.activations.softmax_sampling import SoftmaxSampling
from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.factory import ClassFactoryWrapper
from deep_generative_models.layers.hidden_layers import PartialHiddenLayersFactory

from deep_generative_models.layers.multi_input_layer import MultiInputLayerFactory
from deep_generative_models.layers.multi_output_layer import PartialMultiOutputLayerFactory
from deep_generative_models.layers.single_input_layer import SingleInputLayerFactory
from deep_generative_models.layers.single_output_layer import PartialSingleOutputLayerFactory
from deep_generative_models.losses.autoencoder import AutoEncoderLossFactory
from deep_generative_models.losses.gan import GANGeneratorLossFactory, GANDiscriminatorLossFactory
from deep_generative_models.losses.multi_reconstruction import MultiReconstructionLossFactory
from deep_generative_models.losses.vae import VAELossFactory
from deep_generative_models.losses.wgan import WGANGeneratorLoss, WGANCriticLoss
from deep_generative_models.losses.wgan_gp import WGANCriticLossWithGradientPenaltyFactory

from deep_generative_models.metadata import Metadata

from deep_generative_models.models.autoencoder import SingleVariableAutoEncoderFactory
from deep_generative_models.models.autoencoder import MultiVariableAutoEncoderFactory
from deep_generative_models.models.decoder import SingleOutputDecoderFactory
from deep_generative_models.models.decoder import MultiOutputDecoderFactory
from deep_generative_models.models.denoising_autoencoder import DeNoisingAutoencoderFactory
from deep_generative_models.models.discriminator import DiscriminatorFactory
from deep_generative_models.models.encoder import SingleInputEncoderFactory, MultiInputEncoderFactory
from deep_generative_models.models.generator import SingleOutputGeneratorFactory, MultiOutputGeneratorFactory
from deep_generative_models.models.optimizer import OptimizerFactory
from deep_generative_models.models.vae import VAEFactory


factory_by_name = {
    # my activations
    "GumbelSoftmaxSampling": GumbelSoftmaxSamplingFactory(),
    "SoftmaxSampling": ClassFactoryWrapper(SoftmaxSampling),

    # PyTorch activations (could add more)
    "LeakyReLU": ClassFactoryWrapper(LeakyReLU),
    "ReLU": ClassFactoryWrapper(ReLU),
    "Sigmoid": ClassFactoryWrapper(Sigmoid),
    "Softmax": ClassFactoryWrapper(Softmax),
    "Tanh": ClassFactoryWrapper(Tanh),

    # my losses
    "MultiReconstructionLoss": MultiReconstructionLossFactory(),
    "GANGeneratorLoss": GANGeneratorLossFactory(),
    "GANDiscriminatorLoss": GANDiscriminatorLossFactory(),
    "WGANGeneratorLoss": ClassFactoryWrapper(WGANGeneratorLoss),
    "WGANCriticLoss": ClassFactoryWrapper(WGANCriticLoss),
    "WGANCriticLossWithGradientPenalty": WGANCriticLossWithGradientPenaltyFactory(),

    # PyTorch losses (could add more)
    "BCE": ClassFactoryWrapper(BCELoss),
    "CrossEntropy": ClassFactoryWrapper(CrossEntropyLoss),
    "MSE": ClassFactoryWrapper(MSELoss),

    # PyTorch optimizers (could add more)
    "Adam": OptimizerFactory(Adam),
    "SGD": OptimizerFactory(SGD),
}

# my layers that create other modules
factory_by_name["HiddenLayers"] = PartialHiddenLayersFactory(factory_by_name)
factory_by_name["SingleInputLayer"] = SingleInputLayerFactory(factory_by_name)
factory_by_name["MultiInputLayer"] = MultiInputLayerFactory(factory_by_name)
factory_by_name["SingleOutputLayer"] = PartialSingleOutputLayerFactory(factory_by_name)
factory_by_name["MultiOutputLayer"] = PartialMultiOutputLayerFactory(factory_by_name)

# my losses that create other modules
factory_by_name["AutoEncoderLoss"] = AutoEncoderLossFactory(factory_by_name)
factory_by_name["VAELoss"] = VAELossFactory(factory_by_name)

# my modules that create other modules
factory_by_name["SingleVariableAutoEncoder"] = SingleVariableAutoEncoderFactory(factory_by_name)
factory_by_name["MultiVariableAutoEncoder"] = MultiVariableAutoEncoderFactory(factory_by_name)
factory_by_name["SingleInputEncoder"] = SingleInputEncoderFactory(factory_by_name)
factory_by_name["MultiInputEncoder"] = MultiInputEncoderFactory(factory_by_name)
factory_by_name["SingleOutputDecoder"] = SingleOutputDecoderFactory(factory_by_name)
factory_by_name["MultiOutputDecoder"] = MultiOutputDecoderFactory(factory_by_name)
factory_by_name["SingleOutputGenerator"] = SingleOutputGeneratorFactory(factory_by_name)
factory_by_name["MultiOutputGenerator"] = MultiOutputGeneratorFactory(factory_by_name)
factory_by_name["Discriminator"] = DiscriminatorFactory(factory_by_name, critic=False)
factory_by_name["Critic"] = DiscriminatorFactory(factory_by_name, critic=True)
factory_by_name["SingleVariableDeNoisingAutoencoder"] = DeNoisingAutoencoderFactory(factory_by_name, "SingleVariableAutoEncoder")
factory_by_name["MultiVariableDeNoisingAutoencoder"] = DeNoisingAutoencoderFactory(factory_by_name, "MultiVariableAutoEncoder")
factory_by_name["SingleVariableVAE"] = VAEFactory(factory_by_name, "SingleVariableAutoEncoder")
factory_by_name["MultiVariableVAE"] = VAEFactory(factory_by_name, "MultiVariableAutoEncoder")


def create_architecture(metadata: Metadata, configuration: Configuration) -> Architecture:
    architecture = Architecture()

    # create the dependency nodes
    nodes = set()
    in_edges = dict()
    out_edges = dict()
    for node in configuration.architecture.keys():
        nodes.add(node)
        in_edges[node] = set()
        out_edges[node] = set()

    # create the dependency edges
    nodes_without_out_edges = set()
    for node, child_configuration in configuration.architecture.items():
        factory = factory_by_name[child_configuration.factory]
        dependencies = factory.dependencies(child_configuration.get("arguments", {}))
        if len(dependencies) == 0:
            nodes_without_out_edges.add(node)
        else:
            for other_node in dependencies:
                out_edges[node].add(other_node)  # the node needs the other node
                in_edges[other_node].add(node)  # the other node is needed by the node

    # create modules until the graph is empty (topological sort)
    while len(nodes) > 0:
        # if there are no nodes without out edges there must be a loop
        if len(nodes_without_out_edges) == 0:
            raise Exception("Dependencies cannot be met for modules: {}.".format(", ".join(nodes)))

        # get any node without out edges
        node = nodes_without_out_edges.pop()
        assert len(out_edges[node]) == 0

        # create the module
        child_configuration = configuration.architecture[node]
        factory = factory_by_name[child_configuration.factory]

        architecture[node] = factory.create(architecture,
                                            metadata,
                                            configuration,
                                            child_configuration.get("arguments", {}))

        # while the node has other nodes pointing at him
        while len(in_edges[node]) > 0:
            # remove any incoming edge for the node
            other_node = in_edges[node].pop()
            # remove the outgoing edge for the other node
            out_edges[other_node].remove(node)
            # if the other node has no more dependencies
            if len(out_edges[other_node]) == 0:
                nodes_without_out_edges.add(other_node)

        # remove the node
        nodes.remove(node)
        in_edges.pop(node)
        out_edges.pop(node)

    return architecture
