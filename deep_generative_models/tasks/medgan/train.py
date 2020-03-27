import argparse

from torch import Tensor

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.tasks.gan_with_autoencoder.train import TrainGANWithAutoencoder


class TrainMedGAN(TrainGANWithAutoencoder):

    def sample_fake(self, configuration: Configuration, architecture: Architecture, size: int) -> Tensor:
        fake_code = super(TrainMedGAN, self).sample_fake(configuration, architecture, size)
        return architecture.autoencoder.decode(fake_code)


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="Train MedGAN.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    TrainMedGAN().run(load_configuration(options.configuration))
