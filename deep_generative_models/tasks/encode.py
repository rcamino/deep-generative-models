import torch

from typing import List

import numpy as np

from deep_generative_models.checkpoints import Checkpoints
from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.architecture_factory import create_architecture
from deep_generative_models.gpu import to_cpu_if_was_in_gpu
from deep_generative_models.metadata import load_metadata
from deep_generative_models.tasks.task import Task


class Encode(Task):

    def mandatory_arguments(self) -> List[str]:
        return [
            "metadata",
            "architecture",
            "checkpoint",
            "input",
            "output",
        ]

    def run(self, configuration: Configuration) -> None:
        metadata = load_metadata(configuration.metadata)

        architecture = create_architecture(metadata, load_configuration(configuration.architecture))
        architecture.to_gpu_if_available()

        checkpoints = Checkpoints()
        checkpoint = checkpoints.load(configuration.checkpoint)
        checkpoints.load_states(checkpoint["architecture"], architecture)

        # load the features
        features = torch.from_numpy(np.load(configuration.input))

        # encode
        with torch.no_grad():
            code = architecture.autoencoder.encode(features)["code"]

        # save the code
        code = to_cpu_if_was_in_gpu(code)
        code = code.numpy()
        np.save(configuration.output, code)