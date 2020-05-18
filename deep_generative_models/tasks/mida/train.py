import argparse

import numpy as np

from typing import Dict, List

from torch import Tensor

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.gpu import to_cpu_if_was_in_gpu
from deep_generative_models.imputation.masks import compose_with_mask
from deep_generative_models.losses.multi_reconstruction import MultiReconstructionLoss
from deep_generative_models.metadata import Metadata
from deep_generative_models.post_processing import post_process_discrete
from deep_generative_models.tasks.train import Train, Datasets, Batch


class TrainMIDA(Train):

    def mandatory_architecture_components(self) -> List[str]:
        return [
            "autoencoder",
            "autoencoder_optimizer",
            "reconstruction_loss"
        ]

    def train_epoch(self, configuration: Configuration, metadata: Metadata, architecture: Architecture,
                    datasets: Datasets) -> Dict[str, float]:
        architecture.autoencoder.train()

        # prepare data
        basic_imputation_statistics = datasets.basic_imputation_statistics
        train_datasets = Datasets({"features": datasets.train_features, "missing_mask": datasets.train_missing_mask})
        val_datasets = Datasets({"features": datasets.val_features, "missing_mask": datasets.val_missing_mask})

        # train by batch
        train_loss_by_batch = []

        for batch in self.iterate_datasets(configuration, train_datasets):
            train_loss_by_batch.append(self.train_batch(metadata, architecture, basic_imputation_statistics, batch))

        # loss aggregation
        losses = {"train_reconstruction_mean_loss": np.mean(train_loss_by_batch).item()}

        # validation
        architecture.autoencoder.eval()

        val_loss_by_batch = []

        for batch in self.iterate_datasets(configuration, val_datasets):
            val_loss_by_batch.append(self.val_batch(metadata, architecture, basic_imputation_statistics, batch))

        losses["val_reconstruction_mean_loss"] = np.mean(val_loss_by_batch).item()

        return losses

    @staticmethod
    def initial_imputation(inputs: Tensor, missing_mask: Tensor, basic_imputation_statistics: Tensor) -> Tensor:
        filling_values = basic_imputation_statistics.repeat(len(inputs), 1)
        return compose_with_mask(missing_mask, where_one=filling_values, where_zero=inputs, differentiable=False)

    def train_batch(self, metadata: Metadata, architecture: Architecture, basic_imputation_statistics: Tensor,
                    batch: Batch) -> float:
        architecture.autoencoder_optimizer.zero_grad()

        initially_imputed = self.initial_imputation(batch["features"],
                                                    batch["missing_mask"],
                                                    basic_imputation_statistics)

        outputs = architecture.autoencoder(initially_imputed)

        # must compare with the initial imputation
        # in a real situation the ground truth is not present
        loss = architecture.reconstruction_loss(outputs, initially_imputed)
        loss.backward()

        architecture.autoencoder_optimizer.step()

        loss = to_cpu_if_was_in_gpu(loss)
        return loss.item()

    def val_batch(self, metadata: Metadata, architecture: Architecture, basic_imputation_statistics: Tensor,
                  batch: Batch) -> float:
        initially_imputed = self.initial_imputation(batch["features"],
                                                    batch["missing_mask"],
                                                    basic_imputation_statistics)

        outputs = architecture.autoencoder(initially_imputed)
        reconstructed = post_process_discrete(outputs["reconstructed"], metadata)

        val_reconstruction_loss = MultiReconstructionLoss(metadata)  # force reconstruction loss for validation
        loss = val_reconstruction_loss(reconstructed, batch["features"])  # compare with ground-truth
        loss = to_cpu_if_was_in_gpu(loss)
        return loss.item()


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="Train MIDA.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    TrainMIDA().timed_run(load_configuration(options.configuration))
