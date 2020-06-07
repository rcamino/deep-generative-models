from typing import List, Dict

from torch import Tensor

from deep_generative_models.architecture import Architecture
from deep_generative_models.tasks.impute import Impute


class ImputeWithAutoEncoder(Impute):

    def mandatory_architecture_components(self) -> List[str]:
        return ["autoencoder"]

    def impute(self, architecture: Architecture, batch: Dict[str, Tensor]) -> Tensor:
        return architecture.autoencoder(batch["features"], condition=batch.get("labels"))["reconstructed"]
