import argparse

import numpy as np

from typing import Dict, List, Iterator

from torch import Tensor, FloatTensor

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.gpu import to_gpu_if_available, to_cpu_if_was_in_gpu
from deep_generative_models.metadata import Metadata
from deep_generative_models.post_processing import PostProcessing
from deep_generative_models.pre_processing import PreProcessing
from deep_generative_models.tasks.train import Train, Datasets, Batch


class TrainGAN(Train):

    def mandatory_arguments(self) -> List[str]:
        return super(TrainGAN, self).mandatory_arguments() + ["discriminator_steps", "generator_steps"]

    def mandatory_architecture_arguments(self) -> List[str]:
        return ["noise_size"]

    def mandatory_architecture_components(self) -> List[str]:
        return [
            "generator",
            "discriminator",
            "discriminator_optimizer",
            "discriminator_loss",
            "generator_optimizer",
            "generator_loss"
        ]

    def train_epoch(self, configuration: Configuration, metadata: Metadata, architecture: Architecture,
                    datasets: Datasets, pre_processing: PreProcessing, post_processing: PostProcessing
                    ) -> Dict[str, float]:
        # train
        architecture.generator.train()
        architecture.discriminator.train()

        # prepare to accumulate losses per batch
        losses_by_batch = {"generator": [], "discriminator": []}

        # basic data
        train_datasets = Datasets({"features": datasets.train_features})

        # conditional
        if "conditional" in architecture.arguments:
            train_datasets["labels"] = datasets.train_labels

        # missing mask
        if "train_missing_mask" in datasets:
            train_datasets["missing_mask"] = datasets.train_missing_mask

        # an epoch will stop at any point if there are no more batches
        # it does not matter if there are models with remaining steps
        data_iterator = self.iterate_datasets(configuration, train_datasets)

        while True:
            try:
                losses_by_batch["discriminator"].extend(
                    self.train_discriminator_steps(configuration, metadata, architecture, data_iterator, pre_processing)
                )

                losses_by_batch["generator"].extend(
                    self.train_generator_steps(configuration, metadata, architecture)
                )
            except StopIteration:
                break

        # loss aggregation
        losses = {}

        if configuration.discriminator_steps > 0:
            losses["discriminator_train_mean_loss"] = np.mean(losses_by_batch["discriminator"]).item()

        if configuration.generator_steps > 0:
            losses["generator_train_mean_loss"] = np.mean(losses_by_batch["generator"]).item()

        return losses

    def train_discriminator_steps(self, configuration: Configuration, metadata: Metadata, architecture: Architecture,
                                  batch_iterator: Iterator[Batch], pre_processing: PreProcessing) -> List[float]:
        losses = []
        for _ in range(configuration.discriminator_steps):
            batch = pre_processing.transform(next(batch_iterator))
            loss = self.train_discriminator_step(configuration, metadata, architecture, batch)
            losses.append(loss)
        return losses

    def train_discriminator_step(self, configuration: Configuration, metadata: Metadata, architecture: Architecture,
                                 batch: Batch) -> float:
        # clean previous gradients
        architecture.discriminator_optimizer.zero_grad()

        # generate a batch of fake features with the same size as the real feature batch
        fake_features = self.sample_fake(architecture, len(batch["features"]), condition=batch.get("labels"))
        fake_features = fake_features.detach()  # do not propagate to the generator

        # calculate loss
        loss = architecture.discriminator_loss(architecture,
                                               batch["features"],
                                               fake_features,
                                               condition=batch.get("labels"))

        # calculate gradients
        loss.backward()

        # update the discriminator weights
        architecture.discriminator_optimizer.step()

        # return the loss
        return to_cpu_if_was_in_gpu(loss).item()

    def train_generator_steps(self, configuration: Configuration, metadata: Metadata,
                              architecture: Architecture) -> List[float]:
        losses = []
        for _ in range(configuration.generator_steps):
            loss = self.train_generator_step(configuration, metadata, architecture)
            losses.append(loss)
        return losses

    def train_generator_step(self, configuration: Configuration, metadata: Metadata,
                             architecture: Architecture) -> float:
        # clean previous gradients
        architecture.generator_optimizer.zero_grad()

        # conditional
        if "conditional" in architecture.arguments:
            # for now uniform distribution is used but could be controlled in a different way
            # also this works for both binary and categorical dependent variables
            number_of_conditions = metadata.get_dependent_variable().get_size()
            condition = to_gpu_if_available(FloatTensor(configuration.batch_size).uniform_(0, number_of_conditions))
        # non-conditional
        else:
            condition = None

        # generate a full batch of fake features
        fake_features = self.sample_fake(architecture, configuration.batch_size, condition=condition)

        # calculate loss
        loss = architecture.generator_loss(architecture, fake_features, condition=condition)

        # calculate gradients
        loss.backward()

        # update the generator weights
        architecture.generator_optimizer.step()

        # return the loss
        return to_cpu_if_was_in_gpu(loss).item()

    def sample_fake(self, architecture: Architecture, size: int, **additional_inputs: Tensor) -> Tensor:
        # for now the noise comes from a normal distribution but could be other distribution
        noise = to_gpu_if_available(FloatTensor(size, architecture.arguments.noise_size).normal_())
        return architecture.generator(noise, **additional_inputs)


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="Train GAN.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    TrainGAN().timed_run(load_configuration(options.configuration))
