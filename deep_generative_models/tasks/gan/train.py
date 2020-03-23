import argparse
import time
import torch

import numpy as np

from torch import Tensor
from torch.nn.functional import binary_cross_entropy

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset, Dataset

from typing import Dict

from deep_generative_models.architecture import create_architecture, Architecture
from deep_generative_models.checkpoints import Checkpoints
from deep_generative_models.commandline import create_parent_directories_if_needed
from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.gpu import to_gpu_if_available, to_cpu_if_was_in_gpu
from deep_generative_models.training_logger import TrainingLogger
from deep_generative_models.metadata import load_metadata
from deep_generative_models.models.optimization import create_optimizers, Optimizers
from deep_generative_models.tasks.task import Task


class TrainGAN(Task):

    def run(self, configuration: Configuration) -> None:
        start_time = time.time()
        
        features = torch.from_numpy(np.load(configuration.features))
        data = TensorDataset(features)
        metadata = load_metadata(configuration.metadata)

        architecture = create_architecture(metadata, configuration)
        architecture.to_gpu_if_available()

        optimizers = create_optimizers(architecture, configuration.optimizers)

        checkpoints = Checkpoints(create_parent_directories_if_needed(configuration.checkpoint),
                                  configuration.max_checkpoint_delay)

        if checkpoints.exists():
            checkpoint = checkpoints.load()
        else:
            checkpoint = {
                "architecture": checkpoints.extract_states(architecture),
                "optimizers": checkpoints.extract_states(optimizers),
                "epoch": 0
            }

        logger = TrainingLogger(create_parent_directories_if_needed(configuration.logs), checkpoint["epoch"] > 0)

        for epoch in range(checkpoint["epoch"] + 1, configuration.epochs + 1):
            # train discriminator and generator
            logger.start_timer()
            
            losses = self.train_epoch(configuration, architecture, optimizers, data)
            logger.log(epoch, configuration.epochs, "discriminator", "mean_loss", np.mean(losses["discriminator"]))
            logger.log(epoch, configuration.epochs, "generator", "mean_loss", np.mean(losses["generator"]))

            # update checkpoint
            checkpoint["architecture"] = checkpoints.extract_states(architecture)
            checkpoint["optimizers"] = checkpoints.extract_states(optimizers)
            checkpoint["epoch"] = epoch

            # save checkpoint
            checkpoints.delayed_save(checkpoint)

        # force save of last checkpoint
        checkpoints.save(checkpoint)
        
        # finish
        logger.close()
        print("Total time: {:02f}s".format(time.time() - start_time))

    def train_epoch(self, configuration: Configuration, architecture: Architecture, optimizers: Optimizers,
                    data: Dataset) -> Dict:
        architecture.generator.train()
        architecture.discriminator.train()

        losses = {"generator": [], "discriminator": []}

        more_batches = True
        data_iterator = iter(DataLoader(data, batch_size=configuration.batch_size, shuffle=True))

        while more_batches:
            # train discriminator
            for _ in range(configuration.discriminator_steps):
                # next batch
                try:
                    batch = next(data_iterator)[0]
                    loss = self.train_discriminator(configuration, architecture, optimizers, batch)
                    losses["discriminator"].append(loss)
                except StopIteration:
                    more_batches = False
                    break

            # train generator
            for _ in range(configuration.generator_steps):
                loss = self.train_generator(configuration, architecture, optimizers)
                losses["generator"].append(loss)

        return losses

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
