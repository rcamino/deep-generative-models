import torch

from typing import Any, Dict, List

from torch import Tensor
from torch.nn import Module, Linear

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.metadata import Metadata
from deep_generative_models.component_factory import MultiComponentFactory, ComponentFactory
from deep_generative_models.models.autoencoder import AutoEncoder


class VAE(Module):
    autoencoder: AutoEncoder

    def __init__(self, autoencoder: AutoEncoder, code_size: int) -> None:
        super(VAE, self).__init__()

        self.autoencoder = autoencoder

        self.mu_layer = Linear(code_size, code_size)
        self.log_var_layer = Linear(code_size, code_size)

    def forward(self, inputs: Tensor, **additional_inputs: Tensor) -> Dict[str, Tensor]:
        outputs = self.encode(inputs, **additional_inputs)
        outputs["reconstructed"] = self.decode(outputs["code"], **additional_inputs)
        return outputs

    def encode(self, inputs: Tensor, **additional_inputs: Tensor) -> Dict[str, Tensor]:
        outputs = self.encode_parameters(inputs, **additional_inputs)
        outputs["code"] = self.re_parametrize(outputs["mu"], outputs["log_var"])
        return outputs

    def decode(self, code: Tensor, **additional_inputs: Tensor) -> Tensor:
        return self.autoencoder.decode(code, **additional_inputs)

    def encode_parameters(self, inputs: Tensor, **additional_inputs: Tensor) -> Dict[str, Tensor]:
        split_inputs = self.autoencoder.encode(inputs, **additional_inputs)["code"]
        return {"mu": self.mu_layer(split_inputs), "log_var": self.log_var_layer(split_inputs)}

    @staticmethod
    def re_parametrize(mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return eps * std + mu


class VAEFactory(MultiComponentFactory):
    factory_name: str

    def __init__(self, factory_by_name: Dict[str, ComponentFactory], factory_name: str):
        super(VAEFactory, self).__init__(factory_by_name)
        self.factory_name = factory_name

    def mandatory_architecture_arguments(self) -> List[str]:
        return ["code_size"]

    def mandatory_arguments(self) -> List[str]:
        return ["encoder", "decoder"]

    def create(self, architecture: Architecture, metadata: Metadata, arguments: Configuration) -> Any:
        autoencoder = self.create_other(self.factory_name, architecture, metadata, arguments)
        return VAE(autoencoder, architecture.arguments.code_size)
