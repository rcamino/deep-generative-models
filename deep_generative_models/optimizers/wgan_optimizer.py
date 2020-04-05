from typing import Optional, Callable, Any, List

from torch.optim.optimizer import Optimizer

from deep_generative_models.architecture import Architecture
from deep_generative_models.component_factory import MultiComponentFactory
from deep_generative_models.configuration import Configuration
from deep_generative_models.metadata import Metadata
from deep_generative_models.models.discriminator import Discriminator
from deep_generative_models.optimizers.wrapped_optimizer import WrappedOptimizer


class WGANOptimizer(WrappedOptimizer):
    discriminator: Discriminator
    discriminator_clamp: float

    def __init__(self, optimizer: Optimizer, discriminator: Discriminator, discriminator_clamp: float):
        super(WGANOptimizer, self).__init__(optimizer)
        self.discriminator = discriminator
        self.discriminator_clamp = discriminator_clamp

    def step(self, closure: Optional[Callable[[], float]] = None) -> None:
        for parameter in self.discriminator.parameters():
            parameter.data.clamp_(-self.discriminator_clamp, self.discriminator_clamp)

        super(WGANOptimizer, self).step(closure)


class WGANOptimizerFactory(MultiComponentFactory):

    def dependencies(self, arguments: Configuration) -> List[str]:
        return ["discriminator"]

    def mandatory_arguments(self) -> List[str]:
        return ["optimizer", "discriminator_clamp"]

    def create(self, architecture: Architecture, metadata: Metadata, arguments: Configuration) -> Any:
        optimizer = self.create_other(arguments.optimizer.factory,
                                      architecture,
                                      metadata,
                                      arguments.optimizer.arguments)
        return WGANOptimizer(optimizer, architecture.discriminator, arguments.discriminator_clamp)
