from torch.nn import Module

from deep_generative_models.checkpoint import Checkpoint
from deep_generative_models.architecture import Architecture


def load_module(module: Module, checkpoint: Checkpoint, model_name: str) -> None:
    assert model_name in checkpoint, "'{}' not found in checkpoint.".format(model_name)
    module.load_state_dict(checkpoint[model_name])


def load_architecture(architecture: Architecture, checkpoint: Checkpoint) -> None:
    for module_name, module in architecture.items():
        load_module(module, checkpoint, module_name)
