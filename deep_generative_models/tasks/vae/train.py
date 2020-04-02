import argparse

from deep_generative_models.configuration import load_configuration
from deep_generative_models.tasks.autoencoder.train import TrainAutoEncoder


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="Train VAE.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    TrainAutoEncoder().timed_run(load_configuration(options.configuration))
