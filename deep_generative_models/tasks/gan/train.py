import argparse

import numpy as np

from torch import Tensor, FloatTensor

from torch.utils.data.dataloader import DataLoader

from typing import Dict, List, Iterator

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.gpu import to_gpu_if_available, to_cpu_if_was_in_gpu
from deep_generative_models.metadata import Metadata
from deep_generative_models.tasks.train import Train, Datasets


class TrainGAN(Train):

    def prepare_training(self, configuration: Configuration, metadata: Metadata, architecture: Architecture, datasets: Datasets) -> None:
        architecture.generator.train()
        architecture.discriminator.train()

    def train_epoch(self, configuration: Configuration, metadata: Metadata, architecture: Architecture,
                    datasets: Datasets) -> Dict[str, float]:
        # prepare to accumulate losses per batch
        losses_by_batch = {"generator": [], "discriminator": []}

        # an epoch will stop at any point if there are no more batches
        # it does not matter if there are models with remaining steps
        data_iterator = iter(DataLoader(datasets.train_features, batch_size=configuration.batch_size, shuffle=True))

        while True:
            try:
                losses_by_batch["discriminator"].extend(self.train_discriminator_steps(configuration, architecture, data_iterator))
                losses_by_batch["generator"].extend(self.train_generator_steps(configuration, architecture))
            except StopIteration:
                break

        # loss aggregation
        losses = {"generator_mean_loss": np.mean(losses_by_batch["generator"]).item(),
                  "discriminator_mean_loss": np.mean(losses_by_batch["discriminator"]).item()}

        return losses

    def train_discriminator_steps(self, configuration: Configuration, architecture: Architecture,
                                  data_iterator: Iterator[Tensor]) -> List[float]:
        losses = []
        for _ in range(configuration.discriminator_steps):
            batch = next(data_iterator)
            loss = self.train_discriminator_step(configuration, architecture, batch)
            losses.append(loss)
        return losses

    def train_discriminator_step(self, configuration: Configuration, architecture: Architecture,
                                 real_features: Tensor) -> float:
        # clean previous gradients
        architecture.discriminator_optimizer.zero_grad()

        # generate a batch of fake features with the same size as the real feature batch
        fake_features = self.sample_fake(configuration, architecture, len(real_features))
        fake_features = fake_features.detach()  # do not propagate to the generator

        # calculate loss
        loss = architecture.discriminator_loss(architecture.discriminator, real_features, fake_features)

        # calculate gradients
        loss.backward()

        # update the discriminator weights
        architecture.discriminator_optimizer.step()

        # clamp discriminator parameters (usually for WGAN)
        if "discriminator_clamp" in configuration:
            for parameter in architecture.discriminator.parameters():
                parameter.data.clamp_(-configuration.discriminator_clamp, configuration.discriminator_clamp)

        # return the loss
        return to_cpu_if_was_in_gpu(loss).item()

    def train_generator_steps(self, configuration: Configuration, architecture: Architecture) -> List[float]:
        losses = []
        for _ in range(configuration.generator_steps):
            loss = self.train_generator_step(configuration, architecture)
            losses.append(loss)
        return losses

    def train_generator_step(self, configuration: Configuration, architecture: Architecture) -> float:
        # clean previous gradients
        architecture.generator_optimizer.zero_grad()

        # generate a full batch of fake features
        fake_features = self.sample_fake(configuration, architecture, configuration.batch_size)

        # calculate loss
        loss = architecture.generator_loss(architecture.discriminator, fake_features)

        # calculate gradients
        loss.backward()

        # update the generator weights
        architecture.generator_optimizer.step()

        # return the loss
        return to_cpu_if_was_in_gpu(loss).item()

    def sample_fake(self, configuration: Configuration, architecture: Architecture, size: int) -> Tensor:
        noise = to_gpu_if_available(FloatTensor(size, architecture.arguments.noise_size).normal_())
        return architecture.generator(noise)


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="Train GAN.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    TrainGAN().timed_run(load_configuration(options.configuration))
