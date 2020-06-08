from typing import List, Dict

from torch import Tensor

from deep_generative_models.architecture import Architecture
from deep_generative_models.configuration import Configuration
from deep_generative_models.metadata import Metadata
from deep_generative_models.tasks.impute import Impute


class ImputeWithAutoEncoder(Impute):

    def mandatory_architecture_components(self) -> List[str]:
        return ["autoencoder"]

    def impute(self, configuration: Configuration, metadata: Metadata, architecture: Architecture,
               batch: Dict[str, Tensor]) -> Tensor:
        return architecture.autoencoder(batch["features"], condition=batch.get("labels"))["reconstructed"]
