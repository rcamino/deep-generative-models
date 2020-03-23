import time
import torch

import numpy as np

from torch.utils.data.dataset import TensorDataset, Dataset

from typing import Dict

from deep_generative_models.architecture import create_architecture, Architecture
from deep_generative_models.checkpoints import Checkpoints
from deep_generative_models.commandline import create_parent_directories_if_needed
from deep_generative_models.configuration import Configuration
from deep_generative_models.training_logger import TrainingLogger
from deep_generative_models.metadata import load_metadata
from deep_generative_models.models.optimization import create_optimizers, Optimizers
from deep_generative_models.tasks.task import Task


class Train(Task):

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
            architecture.initialize()

            checkpoint = {
                "architecture": checkpoints.extract_states(architecture),
                "optimizers": checkpoints.extract_states(optimizers),
                "epoch": 0
            }

        logger = TrainingLogger(create_parent_directories_if_needed(configuration.logs), checkpoint["epoch"] > 0)

        for epoch in range(checkpoint["epoch"] + 1, configuration.epochs + 1):
            # train discriminator and generator
            logger.start_timer()

            metrics = self.train_epoch(configuration, architecture, optimizers, data)

            for metric_name, metric_value in metrics.items():
                logger.log(epoch, configuration.epochs, metric_name, metric_value)

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
                    data: Dataset) -> Dict[str, float]:
        raise NotImplementedError
