from torch.optim import Adam, SGD
from torch.optim.optimizer import Optimizer

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.dictionary import Dictionary


class Optimizers(Dictionary[Optimizer]):
    pass


optimizer_classes_by_name = {
    # PyTorch optimizers (could add more)
    "Adam": Adam,
    "SGD": SGD,
}


def create_optimizers(architecture: Architecture, optimizers_configuration: Configuration) -> Optimizers:
    optimizers = Optimizers()
    for optimizer_name, optimizer_configuration in optimizers_configuration.items():
        # extract the module parameters
        parameters = []
        for module_name in optimizer_configuration.modules:
            module = architecture[module_name]
            parameters.extend(module.parameters())
        # create the optimizer
        arguments = optimizer_configuration.get("arguments", default=[], transform_default=False)
        keyword_arguments = optimizer_configuration.get("keyword_arguments", default={}, transform_default=False)
        optimizer_class = optimizer_classes_by_name[optimizer_configuration.factory]
        optimizers[optimizer_name] = optimizer_class(parameters, *arguments, **keyword_arguments)
    return optimizers
