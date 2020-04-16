from typing import Any, List, Dict

from torch.nn import ReLU

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.component_factory import MultiComponentFactory, ComponentFactory
from deep_generative_models.layers.conditional_layer import ConditionalLayer
from deep_generative_models.layers.output_layer import OutputLayerFactory
from deep_generative_models.metadata import Metadata
from deep_generative_models.models.feed_forward import FeedForward


class GeneratorFactory(MultiComponentFactory):

    def mandatory_architecture_arguments(self) -> List[str]:
        return ["noise_size"]

    def optional_arguments(self) -> List[str]:
        return ["hidden_layers"]

    def create(self, architecture: Architecture, metadata: Metadata, arguments: Configuration) -> Any:
        # create input layer
        input_layer = self.create_other("SingleInputLayer", architecture, metadata,
                                        Configuration({"input_size": architecture.arguments.noise_size}))

        # conditional
        if "conditional" in architecture.arguments:
            # wrap the input layer with a conditional layer
            input_layer = ConditionalLayer(input_layer, metadata, **architecture.arguments.conditional)

        # create the hidden layers factory
        hidden_layers_factory = self.create_other("HiddenLayers", architecture, metadata,
                                                  arguments.get("hidden_layers", {}))

        # create the output layer factory
        output_layer_factory = self.create_output_layer_factory(architecture, metadata, arguments)

        # create the generator
        return FeedForward(input_layer, hidden_layers_factory, output_layer_factory, default_hidden_activation=ReLU())

    def create_output_layer_factory(self, architecture: Architecture, metadata: Metadata,
                                    arguments: Configuration) -> OutputLayerFactory:
        raise NotImplementedError


class SingleOutputGeneratorFactory(GeneratorFactory):
    code: bool

    def __init__(self, factory_by_name: Dict[str, ComponentFactory], code: bool = False) -> None:
        super(SingleOutputGeneratorFactory, self).__init__(factory_by_name)
        self.code = code

    def mandatory_architecture_arguments(self) -> List[str]:
        arguments = super(SingleOutputGeneratorFactory, self).mandatory_architecture_arguments()
        if self.code:
            arguments.append("code_size")
        return arguments

    def optional_arguments(self) -> List[str]:
        return ["output_layer"] + super(SingleOutputGeneratorFactory, self).optional_arguments()

    def create_output_layer_factory(self, architecture: Architecture, metadata: Metadata,
                                    arguments: Configuration) -> OutputLayerFactory:
        # override the output layer size
        output_layer_configuration = {}
        if self.code:
            output_layer_configuration["output_size"] = architecture.arguments.code_size
        else:
            output_layer_configuration["output_size"] = metadata.get_num_features()
        # copy activation arguments only if defined
        if "output_layer" in arguments and "activation" in arguments.output_layer:
            output_layer_configuration["activation"] = arguments.output_layer.activation
        # create the output layer factory
        return self.create_other("SingleOutputLayer", architecture, metadata, Configuration(output_layer_configuration))


class MultiOutputGeneratorFactory(GeneratorFactory):

    def mandatory_arguments(self) -> List[str]:
        return ["output_layer"]

    def create_output_layer_factory(self, architecture: Architecture, metadata: Metadata,
                                    arguments: Configuration) -> OutputLayerFactory:
        # create the output layer factory
        return self.create_other("MultiOutputLayer", architecture, metadata, arguments.output_layer)
