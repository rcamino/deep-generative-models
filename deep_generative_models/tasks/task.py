import time

from deep_generative_models.configuration import Configuration
from deep_generative_models.arguments import ArgumentValidator


class Task(ArgumentValidator):

    def timed_run(self, configuration: Configuration) -> None:
        start_time = time.time()
        self.run(configuration)
        print("Total time: {:02f}s".format(time.time() - start_time))

    def run(self, configuration: Configuration) -> None:
        raise NotImplementedError
