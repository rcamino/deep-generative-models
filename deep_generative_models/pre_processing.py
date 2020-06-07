from torch import Tensor

from typing import Optional, Dict

from sklearn.preprocessing import MinMaxScaler

from deep_generative_models.layers.imputation_layer import ImputationLayer


class PreProcessing:
    imputation: Optional[ImputationLayer]

    def __init__(self, imputation: Optional[MinMaxScaler] = None) -> None:
        self.imputation = imputation

    def transform(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if self.imputation is not None:
            assert "missing_mask" in batch
            batch["raw_features"] = batch["features"]
            batch["features"] = self.imputation(batch["features"], batch["missing_mask"])
        return batch
