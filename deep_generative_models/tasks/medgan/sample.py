import argparse

from deep_generative_models.configuration import load_configuration
from deep_generative_models.tasks.gan_with_autoencoder.sample import SampleGANWithAutoEncoder


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="Sample from MedGAN.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    SampleGANWithAutoEncoder().timed_run(load_configuration(options.configuration))
