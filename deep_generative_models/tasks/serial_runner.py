from typing import List

from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.tasks.task import Task


class SerialTaskRunner(Task):

    def mandatory_arguments(self) -> List[str]:
        return ["tasks"]

    def run(self, configuration: Configuration) -> None:
        from deep_generative_models.tasks.runner import TaskRunner  # import here to avoid circular dependency
        task_runner = TaskRunner()
        for child_configuration_path in configuration.tasks:
            task_runner.run(load_configuration(child_configuration_path))
