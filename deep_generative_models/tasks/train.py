import time
import torch

import numpy as np

from torch import Tensor

from typing import Dict

from deep_generative_models.architecture import Architecture
from deep_generative_models.checkpoints import Checkpoints
from deep_generative_models.commandline import create_parent_directories_if_needed
from deep_generative_models.configuration import Configuration
from deep_generative_models.dictionary import Dictionary
from deep_generative_models.factories import create_architecture
from deep_generative_models.training_logger import TrainingLogger
from deep_generative_models.metadata import load_metadata, Metadata
from deep_generative_models.tasks.task import Task


class Datasets(Dictionary[Tensor]):
    pass


class Train(Task):

    def run(self, configuration: Configuration) -> None:
        start_time = time.time()

        datasets = Datasets()
        for dataset_name, dataset_path in configuration.data.items():
            datasets[dataset_name] = torch.from_numpy(np.load(dataset_path))

        metadata = load_metadata(configuration.metadata)

        architecture = create_architecture(metadata, configuration)
        architecture.to_gpu_if_available()

        checkpoints = Checkpoints(create_parent_directories_if_needed(configuration.checkpoint),
                                  configuration.max_checkpoint_delay)

        if checkpoints.exists():
            checkpoint = checkpoints.load()
            checkpoints.load_states(checkpoint["architecture"], architecture)
        else:
            architecture.initialize()

            checkpoint = {
                "architecture": checkpoints.extract_states(architecture),
                "epoch": 0
            }

        logger = TrainingLogger(create_parent_directories_if_needed(configuration.logs), checkpoint["epoch"] > 0)

        for epoch in range(checkpoint["epoch"] + 1, configuration.epochs + 1):
            # train discriminator and generator
            logger.start_timer()

            metrics = self.train_epoch(configuration, metadata, architecture, datasets)

            for metric_name, metric_value in metrics.items():
                logger.log(epoch, configuration.epochs, metric_name, metric_value)

            # update checkpoint
            checkpoint["architecture"] = checkpoints.extract_states(architecture)
            checkpoint["epoch"] = epoch

            # save checkpoint
            checkpoints.delayed_save(checkpoint)

        # force save of last checkpoint
        checkpoints.save(checkpoint)

        # finish
        logger.close()
        print("Total time: {:02f}s".format(time.time() - start_time))

    def train_epoch(self, configuration: Configuration, metadata: Metadata, architecture: Architecture,
                    datasets: Datasets) -> Dict[str, float]:
        raise NotImplementedError
