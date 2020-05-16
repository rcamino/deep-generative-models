import argparse

from torch import Tensor

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import load_configuration
from deep_generative_models.tasks.gan_with_autoencoder.train import TrainGANWithAutoencoder


class TrainMedGAN(TrainGANWithAutoencoder):

    def sample_fake(self, architecture: Architecture, size: int, **additional_inputs: Tensor) -> Tensor:
        fake_code = super(TrainMedGAN, self).sample_fake(architecture, size, **additional_inputs)
        return architecture.autoencoder.decode(fake_code, **additional_inputs)


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="Train MedGAN.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    TrainMedGAN().timed_run(load_configuration(options.configuration))
