import argparse

import numpy as np

from typing import Dict, List, Iterator, Optional

from torch import Tensor, FloatTensor
from torch.utils.data import TensorDataset, DataLoader

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.gpu import to_gpu_if_available, to_cpu_if_was_in_gpu
from deep_generative_models.metadata import Metadata
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
                    datasets: Datasets) -> Dict[str, float]:
        # train
        architecture.generator.train()
        architecture.discriminator.train()

        # prepare to accumulate losses per batch
        losses_by_batch = {"generator": [], "discriminator": []}

        # conditional
        if "conditional" in architecture.arguments:
            train_datasets = TensorDataset(datasets.train_features, datasets.train_labels)
        # non-conditional
        else:
            train_datasets = datasets.train_features

        # an epoch will stop at any point if there are no more batches
        # it does not matter if there are models with remaining steps
        data_iterator = iter(DataLoader(train_datasets, batch_size=configuration.batch_size, shuffle=True))

        while True:
            try:
                losses_by_batch["discriminator"].extend(
                    self.train_discriminator_steps(configuration, metadata, architecture, data_iterator))

                losses_by_batch["generator"].extend(self.train_generator_steps(configuration, metadata, architecture))
            except StopIteration:
                break

        # loss aggregation
        losses = {"generator_train_mean_loss": np.mean(losses_by_batch["generator"]).item(),
                  "discriminator_train_mean_loss": np.mean(losses_by_batch["discriminator"]).item()}

        return losses

    def train_discriminator_steps(self, configuration: Configuration, metadata: Metadata, architecture: Architecture,
                                  batch_iterator: Iterator[Batch]) -> List[float]:
        losses = []
        for _ in range(configuration.discriminator_steps):
            batch = next(batch_iterator)
            loss = self.train_discriminator_step(configuration, metadata, architecture, batch)
            losses.append(loss)
        return losses

    def train_discriminator_step(self, configuration: Configuration, metadata: Metadata, architecture: Architecture,
                                 batch: Batch) -> float:
        # conditional
        if "conditional" in architecture.arguments:
            # use the same conditions for the real and fake features
            real_features, condition = batch
        # non-conditional
        else:
            real_features = batch
            condition = None

        # clean previous gradients
        architecture.discriminator_optimizer.zero_grad()

        # generate a batch of fake features with the same size as the real feature batch
        fake_features = self.sample_fake(architecture, len(real_features), condition=condition)
        fake_features = fake_features.detach()  # do not propagate to the generator

        # calculate loss
        loss = architecture.discriminator_loss(architecture, real_features, fake_features, condition=condition)

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

    def sample_fake(self, architecture: Architecture, size: int, condition: Optional[Tensor] = None) -> Tensor:
        # for now the noise comes from a normal distribution but could be other distribution
        noise = to_gpu_if_available(FloatTensor(size, architecture.arguments.noise_size).normal_())
        return architecture.generator(noise, condition=condition)


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="Train GAN.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    TrainGAN().timed_run(load_configuration(options.configuration))
