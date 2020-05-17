import argparse

from typing import List

from torch.nn import Module

from deep_generative_models.architecture import Architecture
from deep_generative_models.architecture_factory import create_architecture
from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.metadata import load_metadata
from deep_generative_models.tasks.task import Task


def compute_parameter_size(architecture: Architecture) -> int:
    size = 0
    for component in architecture.values():
        if isinstance(component, Module):  # skip optimizers
            for parameter in component.parameters():
                if parameter.requires_grad:
                    size += parameter.numel()
    return size


class ComputeParameterSize(Task):

    def mandatory_arguments(self) -> List[str]:
        return [
            "name",
            "metadata",
            "architecture",
        ]

    def run(self, configuration: Configuration) -> None:
        metadata = load_metadata(configuration.metadata)
        architecture = create_architecture(metadata, load_configuration(configuration.architecture))
        size = compute_parameter_size(architecture)
        self.logger.info("{}: {:d}".format(configuration.name, size))


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="Compute and print the amount of architecture parameters.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    ComputeParameterSize().timed_run(load_configuration(options.configuration))
