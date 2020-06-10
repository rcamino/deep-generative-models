import argparse

from deep_generative_models.configuration import load_configuration
from deep_generative_models.tasks.autoencoder.impute import ImputeWithAutoEncoder


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="Impute with VAE.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    ImputeWithAutoEncoder().timed_run(load_configuration(options.configuration))
