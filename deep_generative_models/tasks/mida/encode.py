import argparse

from deep_generative_models.configuration import load_configuration
from deep_generative_models.tasks.encode import Encode


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="Encode with MIDA.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    Encode().timed_run(load_configuration(options.configuration))
