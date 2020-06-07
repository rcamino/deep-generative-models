import torch

import numpy as np

from typing import List, Dict

from torch import Tensor

from deep_generative_models.architecture import ArchitectureConfigurationValidator, Architecture
from deep_generative_models.architecture_factory import create_architecture, create_component
from deep_generative_models.checkpoints import Checkpoints
from deep_generative_models.configuration import load_configuration, Configuration
from deep_generative_models.gpu import to_cpu_if_was_in_gpu, to_gpu_if_available
from deep_generative_models.imputation.masks import compose_with_mask
from deep_generative_models.metadata import load_metadata
from deep_generative_models.post_processing import load_scale_transform, PostProcessing
from deep_generative_models.pre_processing import PreProcessing
from deep_generative_models.rng import seed_all
from deep_generative_models.tasks.task import Task


class Impute(Task, ArchitectureConfigurationValidator):

    def mandatory_arguments(self) -> List[str]:
        return [
            "features",
            "missing_mask",
            "metadata",
            "architecture",
            "checkpoints",
            "imputation",
            "output"
        ]

    def optional_arguments(self) -> List[str]:
        return super(Impute, self).optional_arguments() + ["seed", "scale_transform"]

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

        # pre-processing
        imputation = create_component(architecture, metadata, configuration.imputation)

        pre_processing = PreProcessing(imputation)

        # post-processing
        if "scale_transform" in configuration:
            scale_transform = load_scale_transform(configuration.scale_transform)
        else:
            scale_transform = None

        post_processing = PostProcessing(metadata, scale_transform)

        # load the features
        features = to_gpu_if_available(torch.from_numpy(np.load(configuration.features)).float())
        missing_mask = to_gpu_if_available(torch.from_numpy(np.load(configuration.missing_mask)).float())

        # initial imputation
        batch = pre_processing.transform({"features": features, "missing_mask": missing_mask})

        # generate the model outputs
        with torch.no_grad():
            output = self.impute(architecture, batch)

        # imputation
        output = compose_with_mask(mask=missing_mask, differentiable=False, where_one=output, where_zero=features)

        # post-process
        output = post_processing.transform(output)

        # save the imputation
        output = to_cpu_if_was_in_gpu(output)
        output = output.numpy()
        np.save(configuration.output, output)

    def impute(self, architecture: Architecture, batch: Dict[str, Tensor]) -> Tensor:
        raise NotImplementedError
