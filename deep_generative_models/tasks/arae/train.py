import argparse

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.metadata import Metadata
from deep_generative_models.tasks.gan_with_autoencoder.train import TrainGANWithAutoencoder
from deep_generative_models.tasks.train import Batch


class TrainARAE(TrainGANWithAutoencoder):

    def train_discriminator_step(self, configuration: Configuration, metadata: Metadata, architecture: Architecture,
                                 batch: Batch) -> float:
        encoded_batch = dict()
        if "conditional" in architecture.arguments:
            encode_result = architecture.autoencoder.encode(batch["features"], condition=batch["labels"])
            encoded_batch["features"] = encode_result["code"]
            encoded_batch["labels"] = batch["labels"]
        else:
            encode_result = architecture.autoencoder.encode(batch["features"])
            encoded_batch["features"] = encode_result["code"]
        return super(TrainARAE, self).train_discriminator_step(configuration, metadata, architecture, encoded_batch)


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="Train ARAE.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    TrainARAE().timed_run(load_configuration(options.configuration))
