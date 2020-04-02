import argparse

from torch.nn import Module

from deep_generative_models.architecture_factory import create_architecture
from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.metadata import load_metadata
from deep_generative_models.tasks.task import Task


class ComputeParameterSize(Task):

    def run(self, configuration: Configuration) -> None:
        metadata = load_metadata(configuration.metadata)
        architecture = create_architecture(metadata, load_configuration(configuration.architecture))

        size = 0
        for component in architecture.values():
            if isinstance(component, Module):  # skip optimizers
                for parameter in component.parameters():
                    if parameter.requires_grad:
                        size += parameter.numel()

        print(configuration.name, size)


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="Compute and print the amount of architecture parameters.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    ComputeParameterSize().timed_run(load_configuration(options.configuration))
