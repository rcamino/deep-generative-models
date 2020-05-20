import torch

from typing import List, Dict

import numpy as np

from deep_generative_models.architecture import Architecture, ArchitectureConfigurationValidator
from deep_generative_models.checkpoints import Checkpoints
from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.architecture_factory import create_architecture
from deep_generative_models.gpu import to_cpu_if_was_in_gpu, to_gpu_if_available
from deep_generative_models.metadata import load_metadata, Metadata
from deep_generative_models.post_processing import load_scale_transform, PostProcessing
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
        return super(Sample, self).optional_arguments() + ["seed", "strategy", "scale_transform"]

    def run(self, configuration: Configuration) -> None:
        seed_all(configuration.get("seed"))

        metadata = load_metadata(configuration.metadata)

        if "scale_transform" in configuration:
            scale_transform = load_scale_transform(configuration.scale_transform)
        else:
            scale_transform = None

        post_processing = PostProcessing(metadata, scale_transform)

        architecture_configuration = load_configuration(configuration.architecture)
        self.validate_architecture_configuration(architecture_configuration)
        architecture = create_architecture(metadata, architecture_configuration)
        architecture.to_gpu_if_available()

        checkpoints = Checkpoints()
        checkpoint = checkpoints.load(configuration.checkpoint)
        checkpoints.load_states(checkpoint["architecture"], architecture)

        samples = []

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

        # this is only to pass less parameters back and forth
        sampler = Sampler(self, configuration, metadata, architecture, post_processing)

        # while more samples are needed
        start = 0
        while start < configuration.sample_size:
            # do not calculate gradients
            with torch.no_grad():
                # sample:
                # the task delegates to the strategy and passes the sampler object to avoid passing even more parameters
                #   the strategy may prepare additional sampling arguments (e.g. condition)
                #   the strategy delegates to the sampler object
                #     the sampler object delegates back to the task adding parameters that it was keeping
                #       the task child class does the actual sampling depending on the model
                #     the sampler object applies post-processing
                #   the strategy may apply filtering to the samples (e.g. rejection)
                # the task finally gets the sample
                batch_samples = strategy.generate_sample(sampler, configuration, metadata)

            # transform back the samples
            batch_samples = to_cpu_if_was_in_gpu(batch_samples)
            batch_samples = batch_samples.numpy()

            # if the batch is not empty
            if len(batch_samples) > 0:
                # do not go further than the desired number of samples
                end = min(start + len(batch_samples), configuration.sample_size)
                # limit the samples taken from the batch based on what is missing
                batch_samples = batch_samples[:min(len(batch_samples), end - start), :]
                # if it is the first batch
                if len(samples) == 0:
                    samples = batch_samples
                # if its not the first batch we have to concatenate
                else:
                    samples = np.concatenate((samples, batch_samples), axis=0)
                # move to next batch
                start = end

        # save the samples
        np.save(configuration.output, samples)

    def generate_sample(self, configuration: Configuration, metadata: Metadata, architecture: Architecture,
                        **additional_inputs: Tensor) -> Tensor:
        raise NotImplementedError


class Sampler:
    sample_task: Sample
    configuration: Configuration
    metadata: Metadata
    architecture: Architecture
    post_processing: PostProcessing

    def __init__(self, sample_task: Sample, configuration: Configuration, metadata: Metadata,
                 architecture: Architecture, post_processing: PostProcessing) -> None:
        self.sample_task = sample_task
        self.configuration = configuration
        self.metadata = metadata
        self.architecture = architecture
        self.post_processing = post_processing

    def generate_sample(self, **additional_inputs: Tensor) -> Tensor:
        # delegate the initial part to the task implementation (that depends on the model)
        sample = self.sample_task.generate_sample(self.configuration,
                                                  self.metadata,
                                                  self.architecture,
                                                  **additional_inputs)
        # apply post-processing
        return self.post_processing.transform(sample)


class SampleStrategy:

    def generate_sample(self, sampler: Sampler, configuration: Configuration, metadata: Metadata) -> Tensor:
        raise NotImplementedError


class DefaultSampleStrategy(SampleStrategy):

    def generate_sample(self, sampler: Sampler, configuration: Configuration, metadata: Metadata) -> Tensor:
        return sampler.generate_sample()


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

    def generate_sample(self, sampler: Sampler, configuration: Configuration, metadata: Metadata) -> Tensor:
        # generate the samples
        samples = sampler.generate_sample()

        # for each desired variable value
        removed_dimensions = 0
        for variable, keep_value in self.keep_values.items():
            # check that the variable is either categorical or binary
            variable_metadata = metadata.get_independent_variable_by_name(variable)
            if not (variable_metadata.is_categorical() or variable_metadata.is_binary()):
                raise Exception("Cannot reject variable '{}' because it has an invalid type.".format(variable))
            # separate the variable
            index = variable_metadata.get_feature_index() - removed_dimensions
            value = samples[:, index:index + variable_metadata.get_size()]
            # for categorical variables we need to transform one-hot encoding into label encoding
            if variable_metadata.is_categorical():
                value = torch.argmax(value, dim=1)
            # reshape value
            value = value.view(-1)
            # keep only the samples with the desired value for that variable
            samples = samples[value == keep_value, :]
            # remove the variable
            left = samples[:, :index]
            right = samples[:, index + variable_metadata.get_size():]
            samples = torch.cat((left, right), dim=1)
            removed_dimensions += variable_metadata.get_size()

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

    def generate_sample(self, sampler: Sampler, configuration: Configuration, metadata: Metadata) -> Tensor:
        condition = to_gpu_if_available(torch.ones(configuration.batch_size, dtype=torch.float) * self.condition)
        return sampler.generate_sample(condition=condition)


strategy_class_by_name = {
    "default": DefaultSampleStrategy,
    "rejection": RejectionSampling,
    "conditional": ConditionalSampling,
}
