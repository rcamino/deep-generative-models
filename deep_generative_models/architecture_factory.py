from typing import Any

from torch.nn import LeakyReLU, ReLU, Sigmoid, Tanh, BCELoss, CrossEntropyLoss, MSELoss, Dropout

from torch.optim import Adam, SGD

from deep_generative_models.activations.gumbel_softmax_sampling import GumbelSoftmaxSamplingFactory
from deep_generative_models.activations.softmax_sampling import SoftmaxSampling
from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.component_factory import ComponentFactoryFromClass
from deep_generative_models.layers.additive_normal_noise import AdditiveNormalNoise
from deep_generative_models.layers.hidden_layers import PartialHiddenLayersFactory
from deep_generative_models.layers.mean_and_modes_imputation_layer import MeanAndModesImputation, MeanAndModesImputationFactory
from deep_generative_models.layers.multi_input_dropout import MultiInputDropoutFactory

from deep_generative_models.layers.multi_input_layer import MultiInputLayerFactory
from deep_generative_models.layers.multi_output_layer import PartialMultiOutputLayerFactory
from deep_generative_models.layers.normal_noise_imputation_layer import NormalNoiseImputation
from deep_generative_models.layers.single_input_layer import SingleInputLayerFactory
from deep_generative_models.layers.single_output_layer import PartialSingleOutputLayerFactory
from deep_generative_models.layers.zero_imputation_layer import ZeroImputation
from deep_generative_models.losses.autoencoder import AutoEncoderLossFactory
from deep_generative_models.losses.gain import GAINDiscriminatorLoss, GAINGeneratorLossFactory
from deep_generative_models.losses.gan import GANGeneratorLoss, GANDiscriminatorLoss
from deep_generative_models.losses.multi_reconstruction import MultiReconstructionLossFactory
from deep_generative_models.losses.rmse import RMSE
from deep_generative_models.losses.vae import VAELossFactory
from deep_generative_models.losses.wgan import WGANGeneratorLoss, WGANCriticLoss
from deep_generative_models.losses.wgan_gp import WGANCriticLossWithGradientPenalty

from deep_generative_models.metadata import Metadata

from deep_generative_models.models.autoencoder import AutoEncoderFactory
from deep_generative_models.models.decoder import SingleOutputDecoderFactory
from deep_generative_models.models.decoder import MultiOutputDecoderFactory
from deep_generative_models.models.denoising_autoencoder import DeNoisingAutoencoderFactory
from deep_generative_models.models.discriminator import DiscriminatorFactory
from deep_generative_models.models.encoder import SingleInputEncoderFactory, MultiInputEncoderFactory
from deep_generative_models.models.gain_discriminator import SingleInputGAINDiscriminatorFactory, MultiInputGAINDiscriminatorFactory
from deep_generative_models.models.gain_generator import MultiVariableGAINGeneratorFactory, SingleVariableGAINGeneratorFactory
from deep_generative_models.models.generator import SingleOutputGeneratorFactory, MultiOutputGeneratorFactory
from deep_generative_models.optimizers.optimizer_factory import OptimizerFactory
from deep_generative_models.models.vae import VAEFactory
from deep_generative_models.optimizers.wgan_optimizer import WGANOptimizerFactory

factory_by_name = {
    # my layers
    "SingleInputLayer": SingleInputLayerFactory(),
    "AdditiveNormalNoise": ComponentFactoryFromClass(AdditiveNormalNoise, ["noise_mean", "noise_std"]),
    "ZeroImputation": ComponentFactoryFromClass(ZeroImputation, ["differentiable"]),
    "NormalNoiseImputation": ComponentFactoryFromClass(NormalNoiseImputation, ["noise_mean", "noise_std", "differentiable"]),
    "MeanAndModesImputation": MeanAndModesImputationFactory(),

    # PyTorch layers (could add more)
    "Dropout": ComponentFactoryFromClass(Dropout, ["p"]),

    # my activations
    "GumbelSoftmaxSampling": GumbelSoftmaxSamplingFactory(),
    "SoftmaxSampling": ComponentFactoryFromClass(SoftmaxSampling),

    # PyTorch activations (could add more)
    "LeakyReLU": ComponentFactoryFromClass(LeakyReLU, ["negative_slope"]),
    "ReLU": ComponentFactoryFromClass(ReLU),
    "Sigmoid": ComponentFactoryFromClass(Sigmoid),
    "Tanh": ComponentFactoryFromClass(Tanh),

    # my losses
    "MultiReconstructionLoss": MultiReconstructionLossFactory(),
    "GANGeneratorLoss": ComponentFactoryFromClass(GANGeneratorLoss, ["smooth_positive_labels"]),
    "GANDiscriminatorLoss": ComponentFactoryFromClass(GANDiscriminatorLoss, ["smooth_positive_labels"]),
    "WGANGeneratorLoss": ComponentFactoryFromClass(WGANGeneratorLoss),
    "WGANCriticLoss": ComponentFactoryFromClass(WGANCriticLoss),
    "WGANCriticLossWithGradientPenalty": ComponentFactoryFromClass(WGANCriticLossWithGradientPenalty, ["weight"]),
    "GAINDiscriminatorLoss": ComponentFactoryFromClass(GAINDiscriminatorLoss),
    "RMSE": ComponentFactoryFromClass(RMSE, ["reduction"]),

    # PyTorch losses (could add more)
    "BCE": ComponentFactoryFromClass(BCELoss, ["weight", "reduction"]),
    "CrossEntropy": ComponentFactoryFromClass(CrossEntropyLoss, ["weight", "reduction"]),
    "MSE": ComponentFactoryFromClass(MSELoss, ["reduction"]),

    # PyTorch optimizers (could add more)
    "Adam": OptimizerFactory(Adam, ["lr", "betas", "eps", "weight_decay", "amsgrad"]),
    "SGD": OptimizerFactory(SGD, ["lr", "momentum", "dampening", "weight_decay", "nesterov"]),
}

# my layers that create other components
factory_by_name["HiddenLayers"] = PartialHiddenLayersFactory(factory_by_name)
factory_by_name["MultiInputLayer"] = MultiInputLayerFactory(factory_by_name)
factory_by_name["SingleOutputLayer"] = PartialSingleOutputLayerFactory(factory_by_name)
factory_by_name["MultiOutputLayer"] = PartialMultiOutputLayerFactory(factory_by_name)
factory_by_name["MultiInputDropout"] = MultiInputDropoutFactory(factory_by_name)

# my losses that create other components
factory_by_name["AutoEncoderLoss"] = AutoEncoderLossFactory(factory_by_name)
factory_by_name["VAELoss"] = VAELossFactory(factory_by_name)
factory_by_name["GAINGeneratorLoss"] = GAINGeneratorLossFactory(factory_by_name)

# my optimizers that create other components
factory_by_name["WGANOptimizer"] = WGANOptimizerFactory(factory_by_name)

# my components that create other components
factory_by_name["SingleVariableAutoEncoder"] = AutoEncoderFactory(factory_by_name, "SingleInputEncoder", "SingleOutputDecoder")
factory_by_name["MultiVariableAutoEncoder"] = AutoEncoderFactory(factory_by_name, "MultiInputEncoder", "MultiOutputDecoder")
factory_by_name["SingleInputEncoder"] = SingleInputEncoderFactory(factory_by_name)
factory_by_name["MultiInputEncoder"] = MultiInputEncoderFactory(factory_by_name)
factory_by_name["SingleOutputDecoder"] = SingleOutputDecoderFactory(factory_by_name)
factory_by_name["MultiOutputDecoder"] = MultiOutputDecoderFactory(factory_by_name)
factory_by_name["SingleOutputGenerator"] = SingleOutputGeneratorFactory(factory_by_name, code=False)
factory_by_name["MultiOutputGenerator"] = MultiOutputGeneratorFactory(factory_by_name)
factory_by_name["Discriminator"] = DiscriminatorFactory(factory_by_name, critic=False, code=False)
factory_by_name["Critic"] = DiscriminatorFactory(factory_by_name, critic=True, code=False)
factory_by_name["SingleVariableDeNoisingAutoencoder"] = DeNoisingAutoencoderFactory(factory_by_name, "SingleVariableAutoEncoder")
factory_by_name["MultiVariableDeNoisingAutoencoder"] = DeNoisingAutoencoderFactory(factory_by_name, "MultiVariableAutoEncoder")
factory_by_name["SingleVariableVAE"] = VAEFactory(factory_by_name, "SingleVariableAutoEncoder")
factory_by_name["MultiVariableVAE"] = VAEFactory(factory_by_name, "MultiVariableAutoEncoder")
factory_by_name["CodeGenerator"] = SingleOutputGeneratorFactory(factory_by_name, code=True)
factory_by_name["CodeDiscriminator"] = DiscriminatorFactory(factory_by_name, critic=False, code=True)
factory_by_name["CodeCritic"] = DiscriminatorFactory(factory_by_name, critic=True, code=True)
factory_by_name["SingleVariableGAINGenerator"] = SingleVariableGAINGeneratorFactory(factory_by_name)
factory_by_name["MultiVariableGAINGenerator"] = MultiVariableGAINGeneratorFactory(factory_by_name)
factory_by_name["SingleInputGAINDiscriminator"] = SingleInputGAINDiscriminatorFactory(factory_by_name)
factory_by_name["MultiInputGAINDiscriminator"] = MultiInputGAINDiscriminatorFactory(factory_by_name)
factory_by_name["SingleVariableMIDA"] = DeNoisingAutoencoderFactory(factory_by_name, "SingleVariableAutoEncoder")
factory_by_name["MultiVariableMIDA"] = DeNoisingAutoencoderFactory(factory_by_name, "MultiVariableAutoEncoder")


def create_architecture(metadata: Metadata, configuration: Configuration) -> Architecture:
    architecture = Architecture(configuration.arguments)

    # create the dependency graph
    # nodes are component names and edges are dependencies between components
    nodes = set()
    in_edges = dict()
    out_edges = dict()
    for node in configuration.components.keys():
        nodes.add(node)
        in_edges[node] = set()
        out_edges[node] = set()

    # create the dependency edges
    nodes_without_out_edges = set()
    for node, component_configuration in configuration.components.items():
        factory = factory_by_name[component_configuration.factory]
        dependencies = factory.dependencies(component_configuration.get("arguments", {}))
        if len(dependencies) == 0:
            nodes_without_out_edges.add(node)
        else:
            for other_node in dependencies:
                out_edges[node].add(other_node)  # the node needs the other node
                in_edges[other_node].add(node)  # the other node is needed by the node

    # create components until the graph is empty (topological sort)
    while len(nodes) > 0:
        # if there are no nodes without out edges there must be a loop
        if len(nodes_without_out_edges) == 0:
            raise Exception("Dependencies cannot be met for components: {}.".format(", ".join(nodes)))

        # get any node without out edges
        node = nodes_without_out_edges.pop()
        assert len(out_edges[node]) == 0

        # create the component
        architecture[node] = create_component(architecture, metadata, configuration.components[node])

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


def create_component(architecture: Architecture, metadata: Metadata, configuration: Configuration) -> Any:
    if "factory" not in configuration:
        raise Exception("Missing factory name while creating component.")

    factory = factory_by_name[configuration.factory]
    arguments = configuration.get("arguments", {})

    return factory.validate_and_create(architecture, metadata, arguments)
