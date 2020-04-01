import argparse

from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.tasks.arae.train import TrainARAE
from deep_generative_models.tasks.autoencoder.train import TrainAutoEncoder
from deep_generative_models.tasks.gan.train import TrainGAN
from deep_generative_models.tasks.medgan.train import TrainMedGAN
from deep_generative_models.tasks.task import Task


task_by_name = {
    "TrainARAE": TrainARAE(),
    "TrainAutoEncoder": TrainAutoEncoder(),
    "TrainGAN": TrainGAN(),
    "TrainMedGAN": TrainMedGAN(),
}


class TaskRunner(Task):

    def run(self, configuration: Configuration) -> None:
        task_by_name[configuration.task].run(configuration.arguments)


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="Run a task.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    TaskRunner().run(load_configuration(options.configuration))