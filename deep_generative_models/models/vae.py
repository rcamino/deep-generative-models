from typing import Any, Dict, List

from torch import Tensor, exp, randn_like
from torch.nn import Module, Linear

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.metadata import Metadata
from deep_generative_models.factory import MultiFactory, Factory
from deep_generative_models.models.autoencoder import AutoEncoder


class VAE(Module):

    autoencoder: AutoEncoder

    def __init__(self, autoencoder: AutoEncoder, split_size: int, code_size: int) -> None:
        super(VAE, self).__init__()

        self.autoencoder = autoencoder

        self.mu_layer = Linear(split_size, code_size)
        self.log_var_layer = Linear(split_size, code_size)

    def forward(self, inputs: Tensor) -> Dict[str, Tensor]:
        outputs = self.encode(inputs)
        outputs["code"] = self.re_parameterize(outputs["mu"], outputs["log_var"])
        outputs["reconstructed"] = self.decode(outputs["code"])
        return outputs

    def encode(self, inputs: Tensor) -> Dict[str, Tensor]:
        outputs = self.autoencoder.encode(inputs)
        return {"mu": self.mu_layer(outputs), "log_var": self.log_var_layer(outputs)}

    @staticmethod
    def re_parameterize(mu: Tensor, log_var: Tensor) -> Tensor:
        std = exp(log_var / 2)
        eps = randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, code: Tensor) -> Tensor:
        return self.autoencoder.decode(code)


class VAEFactory(MultiFactory):
    factory_name: str

    def __init__(self, factory_by_name: Dict[str, Factory], factory_name: str):
        super(VAEFactory, self).__init__(factory_by_name)
        self.factory_name = factory_name

    def mandatory_architecture_arguments(self) -> List[str]:
        return ["split_size", "code_size"]

    def create(self, architecture: Architecture, metadata: Metadata, arguments: Configuration) -> Any:
        autoencoder = self.create_other(self.factory_name, architecture, metadata, arguments)
        return VAE(autoencoder, architecture.arguments.split_size, architecture.arguments.code_size)
