import torch

from typing import List, Dict, Optional

import numpy as np

from deep_generative_models.architecture import Architecture, ArchitectureConfigurationValidator
from deep_generative_models.checkpoints import Checkpoints
from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.architecture_factory import create_architecture
from deep_generative_models.gpu import to_cpu_if_was_in_gpu
from deep_generative_models.metadata import load_metadata, Metadata, VariableMetadata
from deep_generative_models.rng import seed_all
from deep_generative_models.tasks.task import Task

from torch import Tensor


class Sample(Task, ArchitectureConfigurationValidator):

    def mandatory_arguments(self) -> List[str]:
        return [
            "metadata",
            "architecture",
            "checkpoint",
            "output",
            "batch_size",
            "sample_size"
        ]

    def optional_arguments(self) -> List[str]:
        return ["seed", "strategy"]

    def run(self, configuration: Configuration) -> None:
        seed_all(configuration.get("seed"))

        metadata = load_metadata(configuration.metadata)

        architecture_configuration = load_configuration(configuration.architecture)
        self.validate_architecture_configuration(architecture_configuration)
        architecture = create_architecture(metadata, architecture_configuration)
        architecture.to_gpu_if_available()

        checkpoints = Checkpoints()
        checkpoint = checkpoints.load(configuration.checkpoint)
        checkpoints.load_states(checkpoint["architecture"], architecture)

        samples = np.zeros((configuration.sample_size, metadata.get_num_features()), dtype=np.float32)

        # create the strategy if defined
        if "strategy" in configuration:
            # validate strategy name is present
            if "factory" not in configuration.strategy:
                raise Exception("Missing factory name while creating sample strategy.")

            # validate strategy name
            strategy_name = configuration.strategy.factory
            if strategy_name not in strategy_class_by_name:
                raise Exception("Invalid factory name '{}' while creating sample strategy.".format(strategy_name))

            # create the strategy
            strategy_class = strategy_class_by_name[strategy_name]
            strategy = strategy_class(**configuration.strategy.get("arguments", default={}, transform_default=False))

        # use the default strategy
        else:
            strategy = DefaultSampleStrategy()

        # while more samples are needed
        start = 0
        while start < configuration.sample_size:
            # do not calculate gradients
            with torch.no_grad():
                # sample from the model (depending on the strategy)
                batch_samples = strategy.generate_sample(configuration, metadata, architecture, self)

            # transform back the samples
            batch_samples = to_cpu_if_was_in_gpu(batch_samples)
            batch_samples = batch_samples.numpy()

            # if the batch is not empty
            if len(batch_samples) > 0:
                # do not go further than the desired number of samples
                end = min(start + len(batch_samples), configuration.sample_size)
                # limit the samples taken from the batch based on what is missing
                samples[start:end, :] = batch_samples[:min(len(batch_samples), end - start), :]
                # move to next batch
                start = end

        # save the samples
        np.save(configuration.output, samples)

    def generate_sample(self, configuration: Configuration, metadata: Metadata, architecture: Architecture,
                        condition: Optional[Tensor] = None) -> Tensor:
        raise NotImplementedError


class SampleStrategy:

    def generate_sample(self, configuration: Configuration, metadata: Metadata, architecture: Architecture,
                        sampler: Sample) -> Tensor:
        raise NotImplementedError


class DefaultSampleStrategy(SampleStrategy):

    def generate_sample(self, configuration: Configuration, metadata: Metadata, architecture: Architecture,
                        sampler: Sample) -> Tensor:
        return sampler.generate_sample(configuration, metadata, architecture)


class RejectionSampling(SampleStrategy):
    keep_values: Dict[str, float]
    max_iterations: int
    iterations: int
    real_sample_size: int

    def __init__(self, keep_values: Dict[str, float], max_iterations: int):
        self.keep_values = keep_values
        self.max_iterations = max_iterations

        self.iterations = 0
        self.real_sample_size = 0

    def generate_sample(self, configuration: Configuration, metadata: Metadata, architecture: Architecture,
                        sampler: Sample) -> Tensor:
        # generate the samples
        samples = sampler.generate_sample(configuration, metadata, architecture)

        # for each desired variable value
        for variable, keep_value in self.keep_values.items():
            # check that the variable is either categorical or binary
            variable_metadata = metadata.get_independent_variable_by_name(variable)
            if not (variable_metadata.is_categorical() or variable_metadata.is_binary()):
                raise Exception("Cannot reject variable '{}' because it has an invalid type.".format(variable))
            # separate the variable
            index = variable_metadata.get_feature_index()
            value = samples[:, index:index + variable_metadata.get_size()]
            # for categorical variables we need to transform one-hot encoding into label encoding
            if variable_metadata.is_categorical():
                value = torch.argmax(value, dim=1)
            # get the relative id of the value that should be kept
            keep_value_id = variable_metadata.get_index_from_value(keep_value)
            # keep only the samples with the desired value for that variable
            samples = samples[value == keep_value_id, :]

        # recalculate after filtering
        real_batch_size = len(samples)
        self.real_sample_size += real_batch_size

        # if there was a change
        if real_batch_size > 0:
            # restart iterations
            self.iterations = 0
        # if there was no change count another iteration
        else:
            # try again
            self.iterations += 1

            if self.iterations >= self.max_iterations:
                raise Exception("Reached maximum number of iterations with {:d} samples.".format(self.real_sample_size))

        return samples


class ConditionalSampling(SampleStrategy):
    condition: int

    def __init__(self, condition: int):
        self.condition = condition

    def generate_sample(self, configuration: Configuration, metadata: Metadata, architecture: Architecture,
                        sampler: Sample) -> Tensor:
        condition = torch.ones(configuration.batch_size, dtype=torch.float) * self.condition
        return sampler.generate_sample(configuration, metadata, architecture, condition=condition)


strategy_class_by_name = {
    "default": DefaultSampleStrategy,
    "rejection": RejectionSampling,
    "conditional": ConditionalSampling,
}
