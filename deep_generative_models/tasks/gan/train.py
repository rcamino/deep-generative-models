import argparse

import numpy as np

from torch import Tensor, FloatTensor

from torch.utils.data.dataloader import DataLoader

from typing import Dict

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.gpu import to_gpu_if_available, to_cpu_if_was_in_gpu
from deep_generative_models.metadata import Metadata
from deep_generative_models.tasks.autoencoder.train import TrainAutoEncoder
from deep_generative_models.tasks.gan.strategy import create_gan_strategy, GANStrategy
from deep_generative_models.tasks.train import Train, Datasets


class TrainGAN(Train):

    def train_epoch(self, configuration: Configuration, metadata: Metadata, architecture: Architecture,
                    datasets: Datasets) -> Dict[str, float]:
        # put models in training mode (this should reach the autoencoder if present)
        architecture.generator.train()
        architecture.discriminator.train()

        # prepare to accumulate losses per batch
        losses_by_batch = {"generator": [], "discriminator": []}

        # how the autoencoder interacts with the GAN
        # be aware that sometimes even if the autoencoder is not trained it might be used (like in MedGAN)
        strategy_name = configuration.get("gan_strategy", "VanillaGAN")
        strategy = create_gan_strategy(architecture, strategy_name)

        # prepare autoencoder training if needed
        if "autoencoder_steps" in configuration:
            assert strategy_name != "VanillaGAN"
            autoencoder_steps = configuration.autoencoder_steps
            autoencoder_train_task = TrainAutoEncoder()
            losses_by_batch["autoencoder"] = []
        # no autoencoder training
        else:
            autoencoder_steps = 0
            autoencoder_train_task = None

        # an epoch will stop at any point if there are no more batches
        # it does not matter if there are models with remaining steps
        more_batches = True
        data_iterator = iter(DataLoader(datasets.train_features, batch_size=configuration.batch_size, shuffle=True))

        while more_batches:
            # train autoencoder (optional)
            for _ in range(autoencoder_steps):
                # next batch
                try:
                    batch = next(data_iterator)
                    loss = autoencoder_train_task.train_batch(architecture, batch)
                    losses_by_batch["autoencoder"].append(loss)
                except StopIteration:
                    more_batches = False
                    break

            # train discriminator
            for _ in range(configuration.discriminator_steps):
                # next batch
                try:
                    batch = next(data_iterator)
                    loss = self.train_discriminator(configuration, architecture, strategy, batch)
                    losses_by_batch["discriminator"].append(loss)
                except StopIteration:
                    more_batches = False
                    break

            # train generator
            for _ in range(configuration.generator_steps):
                loss = self.train_generator(configuration, architecture, strategy)
                losses_by_batch["generator"].append(loss)

        # loss aggregation
        losses = {"generator_mean_loss": np.mean(losses_by_batch["generator"]).item(),
                  "discriminator_mean_loss": np.mean(losses_by_batch["discriminator"]).item()}

        # add autoencoder loss (optional)
        if autoencoder_steps > 0:
            losses["autoencoder_mean_loss"] = np.mean(losses_by_batch["autoencoder"]).item()

        return losses

    @staticmethod
    def train_discriminator(configuration: Configuration, architecture: Architecture, strategy: GANStrategy,
                            real_features: Tensor) -> float:
        # clean previous gradients
        architecture.discriminator_optimizer.zero_grad()

        # wrap real features (in case an autoencoder is used)
        real_features = strategy.wrap_real_features(real_features)

        # generate a batch of fake features with the same size as the real feature batch
        noise = to_gpu_if_available(FloatTensor(len(real_features), configuration.noise_size).normal_())
        generator_outputs = architecture.generator(noise)
        fake_features = strategy.wrap_generator_outputs(generator_outputs)
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

    @staticmethod
    def train_generator(configuration: Configuration, architecture: Architecture, strategy: GANStrategy) -> float:
        # clean previous gradients
        architecture.generator_optimizer.zero_grad()

        # generate a full batch of fake features
        noise = to_gpu_if_available(FloatTensor(configuration.batch_size, configuration.noise_size).normal_())
        generator_outputs = architecture.generator(noise)
        fake_features = strategy.wrap_generator_outputs(generator_outputs)

        # calculate loss
        loss = architecture.generator_loss(architecture.discriminator, fake_features)

        # calculate gradients
        loss.backward()

        # update the generator weights
        architecture.generator_optimizer.step()

        # return the loss
        return to_cpu_if_was_in_gpu(loss).item()


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="Train GAN.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    TrainGAN().run(load_configuration(options.configuration))
