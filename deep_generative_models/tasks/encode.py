import torch

from typing import List

import numpy as np

from deep_generative_models.architecture import ArchitectureConfigurationValidator
from deep_generative_models.checkpoints import Checkpoints
from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.architecture_factory import create_architecture
from deep_generative_models.gpu import to_cpu_if_was_in_gpu, to_gpu_if_available
from deep_generative_models.metadata import load_metadata
from deep_generative_models.rng import seed_all
from deep_generative_models.tasks.task import Task


class Encode(Task, ArchitectureConfigurationValidator):

    def mandatory_arguments(self) -> List[str]:
        return [
            "metadata",
            "architecture",
            "checkpoint",
            "features",
            "output",
        ]

    def optional_arguments(self) -> List[str]:
        return super(Encode, self).optional_arguments() + ["seed", "labels"]

    def mandatory_architecture_components(self) -> List[str]:
        return ["autoencoder"]

    def run(self, configuration: Configuration) -> None:
        seed_all(configuration.get("seed"))

        metadata = load_metadata(configuration.metadata)

        architecture_configuration = load_configuration(configuration.architecture)
        self.validate_architecture_configuration(architecture_configuration)
        architecture = create_architecture(metadata, architecture_configuration)
        architecture.to_gpu_if_available()

        checkpoints = Checkpoints()
        checkpoint = checkpoints.load(configuration.checkpoint)
        if "best_architecture" in checkpoint:
            checkpoints.load_states(checkpoint["best_architecture"], architecture)
        else:
            checkpoints.load_states(checkpoint["architecture"], architecture)

        # load the features
        features = to_gpu_if_available(torch.from_numpy(np.load(configuration.features)).float())

        # conditional
        if "labels" in configuration:
            condition = to_gpu_if_available(torch.from_numpy(np.load(configuration.labels)).float())
        else:
            condition = None

        # encode
        with torch.no_grad():
            code = architecture.autoencoder.encode(features, condition=condition)["code"]

        # save the code
        code = to_cpu_if_was_in_gpu(code)
        code = code.numpy()
        np.save(configuration.output, code)
