import argparse

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.metadata import Metadata
from deep_generative_models.tasks.gan_with_autoencoder.train import TrainGANWithAutoencoder
from deep_generative_models.tasks.train import Batch


class TrainARAE(TrainGANWithAutoencoder):

    def train_discriminator_step(self, configuration: Configuration, metadata: Metadata, architecture: Architecture,
                                 batch: Batch) -> float:
        if "conditional" in architecture.arguments:
            real_features, real_labels = batch
            real_code = architecture.autoencoder.encode(real_features)["code"]
            encoded_batch = real_code, real_labels
            return super(TrainARAE, self).train_discriminator_step(configuration, metadata, architecture, encoded_batch)
        else:
            real_features = batch
            real_code = architecture.autoencoder.encode(real_features)["code"]
            return super(TrainARAE, self).train_discriminator_step(configuration, metadata, architecture, real_code)


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="Train ARAE.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    TrainARAE().timed_run(load_configuration(options.configuration))
