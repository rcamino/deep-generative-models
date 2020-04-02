import argparse

from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.tasks.arae.train import TrainARAE
from deep_generative_models.tasks.autoencoder.train import TrainAutoEncoder
from deep_generative_models.tasks.gan.sample import SampleGAN
from deep_generative_models.tasks.gan.train import TrainGAN
from deep_generative_models.tasks.gan_with_autoencoder.sample import SampleGANWithAutoEncoder
from deep_generative_models.tasks.medgan.train import TrainMedGAN
from deep_generative_models.tasks.task import Task
from deep_generative_models.tasks.vae.sample import SampleVAE

task_by_name = {
    # train
    "TrainARAE": TrainARAE(),
    "TrainAutoEncoder": TrainAutoEncoder(),
    "TrainGAN": TrainGAN(),
    "TrainMedGAN": TrainMedGAN(),

    # sample
    "SampleGAN": SampleGAN(),
    "SampleGANWithAutoEncoder": SampleGANWithAutoEncoder(),
    "SampleVAE": SampleVAE(),
}


class TaskRunner(Task):

    def run(self, configuration: Configuration) -> None:
        task = task_by_name[configuration.task]
        task.validate_arguments(configuration.arguments)
        task.run(configuration.arguments)


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="Run a task.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    TaskRunner().timed_run(load_configuration(options.configuration))
