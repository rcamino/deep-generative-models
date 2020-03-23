import argparse
import torch

import numpy as np

from torch import Tensor
from torch.nn.functional import binary_cross_entropy

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from typing import Dict

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.gpu import to_gpu_if_available, to_cpu_if_was_in_gpu
from deep_generative_models.metadata import Metadata
from deep_generative_models.tasks.train import Train
from deep_generative_models.models.optimization import Optimizers


class TrainGAN(Train):

    def train_epoch(self, configuration: Configuration, metadata: Metadata, architecture: Architecture,
                    optimizers: Optimizers, data: Dataset) -> Dict[str, float]:
        architecture.generator.train()
        architecture.discriminator.train()

        loss_by_batch = {"generator": [], "discriminator": []}

        more_batches = True
        data_iterator = iter(DataLoader(data, batch_size=configuration.batch_size, shuffle=True))

        while more_batches:
            # train discriminator
            for _ in range(configuration.discriminator_steps):
                # next batch
                try:
                    batch = next(data_iterator)[0]
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
                            batch: Tensor) -> float:
        # generate input and label vectors
        label_zeros = torch.zeros(len(batch))
        smooth_label_ones = torch.FloatTensor(len(batch)).uniform_(0.9, 1)
        noise = torch.FloatTensor(len(batch), configuration.noise_size).normal_()  # match current batch size
        label_zeros, smooth_label_ones, noise = to_gpu_if_available(label_zeros, smooth_label_ones, noise)

        optimizers.discriminator.zero_grad()

        # first train the discriminator only with real data
        real_predictions = architecture.discriminator(batch)
        real_loss = binary_cross_entropy(real_predictions, smooth_label_ones)
        real_loss.backward()

        # then train the discriminator only with fake data
        fake_features = architecture.generator(noise)
        fake_features = fake_features.detach()  # do not propagate to the generator
        fake_predictions = architecture.discriminator(fake_features)
        fake_loss = binary_cross_entropy(fake_predictions, label_zeros)
        fake_loss.backward()

        # finally update the discriminator weights
        # using two separated batches is another trick to improve GAN training
        optimizers.discriminator.step()

        # calculate total loss
        loss = real_loss + fake_loss
        loss = to_cpu_if_was_in_gpu(loss)
        return loss.item()

    @staticmethod
    def train_generator(configuration: Configuration, architecture: Architecture, optimizers: Optimizers) -> float:
        # generate input and label vectors
        noise = torch.FloatTensor(configuration.batch_size, configuration.noise_size).normal_()  # full batch
        smooth_label_ones = torch.FloatTensor(len(noise)).uniform_(0.9, 1)
        noise, smooth_label_ones = to_gpu_if_available(noise, smooth_label_ones)

        optimizers.generator.zero_grad()

        features = architecture.generator(noise)
        predictions = architecture.discriminator(features)

        loss = binary_cross_entropy(predictions, smooth_label_ones)
        loss.backward()

        optimizers.generator.step()

        loss = to_cpu_if_was_in_gpu(loss)
        return loss.item()


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="Train GAN.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    TrainGAN().run(load_configuration(options.configuration))
