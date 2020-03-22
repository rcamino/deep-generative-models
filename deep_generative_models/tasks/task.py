from deep_generative_models.configuration import Configuration


class Task:

    def run(self, configuration: Configuration) -> None:
        raise NotImplementedError
