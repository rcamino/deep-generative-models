import time

from deep_generative_models.configuration import Configuration


class Task:

    def timed_run(self, configuration: Configuration) -> None:
        start_time = time.time()
        self.run(configuration)
        print("Total time: {:02f}s".format(time.time() - start_time))

    def run(self, configuration: Configuration) -> None:
        raise NotImplementedError
