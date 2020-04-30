import argparse

from typing import List, Any

from augmentation.task import AugmentationTask
from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.tasks.multiprocess_runner import MultiProcessTaskWorker, MultiProcessTaskRunner


class AugmentationWorker(MultiProcessTaskWorker):

    @classmethod
    def output_fields(cls) -> List[str]:
        return [
            "case",
            "name",
            "train_pr_auc_mean",
            "train_pr_auc_std",
            "test_pr_auc_mean",
            "test_pr_auc_std",
            "train_f1_mean",
            "train_f1_std",
            "test_f1_mean",
            "test_f1_std",
            "under_sampling",
            "under_sampling_ratio",
            "over_sampling",
            "over_sampling_ratio",
            "over_sampling_sample_path",
            "has_error",
            "error",
            "time",
        ]

    def process(self, inputs: Any) -> None:
        AugmentationTask(self).run(Configuration(inputs))


def main():
    options_parser = argparse.ArgumentParser(description="Run augmentation tasks.")
    options_parser.add_argument("configuration", type=str, help="Configuration json file.")
    options = options_parser.parse_args()

    MultiProcessTaskRunner(AugmentationWorker).timed_run(load_configuration(options.configuration_file))


if __name__ == '__main__':
    main()
