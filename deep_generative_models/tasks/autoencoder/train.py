import argparse

import numpy as np

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from typing import Dict

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.gpu import to_cpu_if_was_in_gpu
from deep_generative_models.metadata import Metadata
from deep_generative_models.models.optimization import Optimizers
from deep_generative_models.tasks.train import Train


class TrainAutoEncoder(Train):

    def train_epoch(self, configuration: Configuration, metadata: Metadata, architecture: Architecture,
                    optimizers: Optimizers, data: Dataset) -> Dict[str, float]:
        architecture.autoencoder.train()

        loss_by_batch = []

        for batch, in DataLoader(data, batch_size=configuration.batch_size, shuffle=True):
            optimizers.autoencoder.zero_grad()

            _, reconstructed = architecture.autoencoder(batch)

            loss = architecture.reconstruction_loss(reconstructed, batch)
            loss.backward()

            optimizers.autoencoder.step()

            loss = to_cpu_if_was_in_gpu(loss)
            loss_by_batch.append(loss.item())

        return {"reconstruction_mean_loss": np.mean(loss_by_batch).item()}


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="Train AutoEncoder.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    TrainAutoEncoder().run(load_configuration(options.configuration))
