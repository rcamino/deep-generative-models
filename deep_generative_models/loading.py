from torch.nn import Module

from typing import Optional

from deep_generative_models.type_aliases import Checkpoint, Architecture


def load_module(module: Module, checkpoint: Optional[Checkpoint], model_name: str) -> None:
    if checkpoint is not None:
        assert model_name in checkpoint, "'{}' not found in checkpoint.".format(model_name)
        module.load_state_dict(checkpoint[model_name])


def load_modules(modules: Architecture, checkpoint: Optional[Checkpoint]) -> None:
    for module_name, module in modules.items():
        load_module(module, checkpoint, module_name)
