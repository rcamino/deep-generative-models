from torch import Tensor

from typing import Optional, Dict

from deep_generative_models.layers.imputation_layer import ImputationLayer


class PreProcessing:
    imputation: Optional[ImputationLayer]

    def __init__(self, imputation: Optional[ImputationLayer] = None) -> None:
        self.imputation = imputation

    def transform(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if "missing_mask" in batch:
            assert self.imputation is not None, "There are missing values but imputation not available."
            batch["raw_features"] = batch["features"]
            batch["features"] = self.imputation(batch["features"], batch["missing_mask"])
        return batch
