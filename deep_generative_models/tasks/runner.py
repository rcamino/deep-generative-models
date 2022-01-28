import argparse

from deep_generative_models.arguments import InvalidArgument, MissingArgument
from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.imputation.compute_mean_and_modes_task import ComputeMeansAndModes
from deep_generative_models.imputation.means_and_modes_imputation_task import MeansAndModesImputation
from deep_generative_models.imputation.miss_forest_imputation_task import MissForestImputation
from deep_generative_models.imputation.noise_imputation_task import NormalNoiseImputation
from deep_generative_models.imputation.train_miss_forest_task import TrainMissForest
from deep_generative_models.imputation.zero_imputation_task import ZeroImputation
from deep_generative_models.tasks.arae.train import TrainARAE
from deep_generative_models.tasks.autoencoder.impute import ImputeWithAutoEncoder
from deep_generative_models.tasks.autoencoder.train import TrainAutoEncoder
from deep_generative_models.tasks.encode import Encode
from deep_generative_models.tasks.gain.impute import ImputeWithGAIN
from deep_generative_models.tasks.gain.train import TrainGAIN
from deep_generative_models.tasks.gan.iterative_imputation import GANIterativeImputation
from deep_generative_models.tasks.gan.sample import SampleGAN
from deep_generative_models.tasks.gan.train import TrainGAN
from deep_generative_models.tasks.gan_with_autoencoder.sample import SampleGANWithAutoEncoder
from deep_generative_models.imputation.generate_missing_mask_task import GenerateMissingMask
from deep_generative_models.tasks.medgan.train import TrainMedGAN
from deep_generative_models.tasks.multiprocess_runner import MultiProcessTaskRunner, TaskRunnerWorker
from deep_generative_models.tasks.serial_runner import SerialTaskRunner
from deep_generative_models.tasks.task import Task
from deep_generative_models.tasks.vae.sample import SampleVAE


task_by_name = {
    # train
    "TrainARAE": TrainARAE(),
    "TrainAutoEncoder": TrainAutoEncoder(),
    "TrainDeNoisingAutoEncoder": TrainAutoEncoder(),
    "TrainGAN": TrainGAN(),
    "TrainMedGAN": TrainMedGAN(),
    "TrainVAE": TrainAutoEncoder(),
    "TrainGAIN": TrainGAIN(),
    "TrainMIDA": TrainAutoEncoder(),

    # sample
    "SampleARAE": SampleGANWithAutoEncoder(),
    "SampleGAN": SampleGAN(),
    "SampleMedGAN": SampleGANWithAutoEncoder(),
    "SampleVAE": SampleVAE(),

    # encode (and some silly aliases)
    "Encode": Encode(),
    "EncodeWithAutoEncoder": Encode(),
    "EncodeWithDeNoisingAutoEncoder": Encode(),
    "EncodeWithVAE": Encode(),
    "EncodeWithMIDA": Encode(),

    # imputation
    "GenerateMissingMask": GenerateMissingMask(),
    "ComputeMeansAndModes": ComputeMeansAndModes(),
    "MeansAndModesImputation": MeansAndModesImputation(),
    "NormalNoiseImputation": NormalNoiseImputation(),
    "ZeroImputation": ZeroImputation(),
    "TrainMissForest": TrainMissForest(),
    "MissForestImputation": MissForestImputation(),
    "ImputeWithMIDA": ImputeWithAutoEncoder(),
    "ImputeWithVAE": ImputeWithAutoEncoder(),
    "ImputeWithGAIN": ImputeWithGAIN(),
    "GANIterativeImputation": GANIterativeImputation(),

    # runners
    "SerialTaskRunner": SerialTaskRunner(),
    "MultiProcessTaskRunner": MultiProcessTaskRunner(TaskRunnerWorker)
}


class TaskRunner(Task):

    def run(self, configuration: Configuration) -> None:
        task = task_by_name[configuration.task]

        try:
            task.validate_arguments(configuration.arguments)
        except MissingArgument as e:
            raise Exception("Missing argument '{}' while running task '{}'".format(e.name, configuration.task))
        except InvalidArgument as e:
            raise Exception("Invalid argument '{}' while running task '{}'".format(e.name, configuration.task))

        task.run(configuration.arguments)


if __name__ == '__main__':
    options_parser = argparse.ArgumentParser(description="Run a task.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    TaskRunner().timed_run(load_configuration(options.configuration))
