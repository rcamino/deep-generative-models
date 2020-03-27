import argparse

from torch import Tensor

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.tasks.gan_with_autoencoder.train import TrainGANWithAutoencoder


class TrainARAE(TrainGANWithAutoencoder):

    def train_discriminator_step(self, configuration: Configuration, architecture: Architecture,
                                 real_features: Tensor) -> float:
        real_code = architecture.autoencoder.encode(real_features)
        return super(TrainARAE, self).train_discriminator_step(configuration, architecture, real_code)


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="Train ARAE.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    TrainARAE().run(load_configuration(options.configuration))
