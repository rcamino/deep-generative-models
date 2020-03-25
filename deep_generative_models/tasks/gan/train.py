import argparse

import numpy as np

from torch import Tensor, FloatTensor

from torch.utils.data.dataloader import DataLoader

from typing import Dict

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.gpu import to_gpu_if_available, to_cpu_if_was_in_gpu
from deep_generative_models.metadata import Metadata
from deep_generative_models.tasks.train import Train, Datasets
from deep_generative_models.models.optimization import Optimizers


class TrainGAN(Train):

    def train_epoch(self, configuration: Configuration, metadata: Metadata, architecture: Architecture,
                    optimizers: Optimizers, datasets: Datasets) -> Dict[str, float]:
        architecture.generator.train()
        architecture.discriminator.train()

        loss_by_batch = {"generator": [], "discriminator": []}

        more_batches = True
        data_iterator = iter(DataLoader(datasets.train_features, batch_size=configuration.batch_size, shuffle=True))

        while more_batches:
            # train discriminator
            for _ in range(configuration.discriminator_steps):
                # next batch
                try:
                    batch = next(data_iterator)
                    loss = self.train_discriminator(configuration, architecture, optimizers, batch)
                    loss_by_batch["discriminator"].append(loss)
                except StopIteration:
                    more_batches = False
                    break

            # train generator
            for _ in range(configuration.generator_steps):
                loss = self.train_generator(configuration, architecture, optimizers)
                loss_by_batch["generator"].append(loss)

        return {"generator_mean_loss": np.mean(loss_by_batch["generator"]).item(),
                "discriminator_mean_loss": np.mean(loss_by_batch["discriminator"]).item()}

    @staticmethod
    def train_discriminator(configuration: Configuration, architecture: Architecture, optimizers: Optimizers,
                            real_features: Tensor) -> float:
        # clean previous gradients
        optimizers.discriminator.zero_grad()

        # generate a batch of fake features with the same size as the real feature batch
        noise = to_gpu_if_available(FloatTensor(len(real_features), configuration.noise_size).normal_())
        fake_features = architecture.generator(noise)
        fake_features = fake_features.detach()  # do not propagate to the generator

        # calculate loss
        loss = architecture.discriminator_loss(architecture.discriminator, real_features, fake_features)

        # calculate gradients
        loss.backward()

        # update the discriminator weights
        optimizers.discriminator.step()

        # clamp discriminator parameters (usually for WGAN)
        if "discriminator_clamp" in configuration:
            for parameter in architecture.discriminator.parameters():
                parameter.data.clamp_(-configuration.discriminator_clamp, configuration.discriminator_clamp)

        # return the loss
        return to_cpu_if_was_in_gpu(loss).item()

    @staticmethod
    def train_generator(configuration: Configuration, architecture: Architecture, optimizers: Optimizers) -> float:
        # clean previous gradients
        optimizers.generator.zero_grad()

        # generate a full batch of fake features
        noise = to_gpu_if_available(FloatTensor(configuration.batch_size, configuration.noise_size).normal_())
        fake_features = architecture.generator(noise)

        # calculate loss
        loss = architecture.generator_loss(architecture.discriminator, fake_features)

        # calculate gradients
        loss.backward()

        # update the generator weights
        optimizers.generator.step()

        # return the loss
        return to_cpu_if_was_in_gpu(loss).item()


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="Train GAN.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    TrainGAN().run(load_configuration(options.configuration))
