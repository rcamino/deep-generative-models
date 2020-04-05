from typing import Optional, Callable

from torch.optim.optimizer import Optimizer


class WrappedOptimizer:
    optimizer: Optimizer

    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer

    def state_dict(self) -> dict:
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.optimizer.load_state_dict(state_dict)

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def step(self, closure: Optional[Callable[[], float]] = None) -> None:
        self.optimizer.step(closure)

    def add_param_group(self, param_group: dict) -> None:
        self.optimizer.add_param_group(param_group)
