import pickle
import torch

from typing import Optional

from sklearn.preprocessing import MinMaxScaler

from torch import Tensor
from torch.nn.functional import one_hot

from deep_generative_models.gpu import to_cpu_if_was_in_gpu
from deep_generative_models.metadata import Metadata


def load_scale_transform(path: str) -> MinMaxScaler:
    with open(path, "rb") as scale_transform_file:
        return pickle.load(scale_transform_file)


class PostProcessing:
    metadata: Metadata
    scale_transform: Optional[MinMaxScaler]

    def __init__(self, metadata: Metadata, scale_transform: Optional[MinMaxScaler] = None) -> None:
        self.metadata = metadata
        self.scale_transform = scale_transform

    def transform(self, sample: Tensor) -> Tensor:
        return self.continuous_transform(self.discrete_transform(sample))

    def continuous_transform(self, sample: Tensor) -> Tensor:
        if self.scale_transform is not None:
            # TODO: I don't like going back and forth from numpy and torch
            numpy_sample = to_cpu_if_was_in_gpu(sample.detach()).numpy()
            unscaled = self.scale_transform.inverse_transform(numpy_sample)
            return torch.from_numpy(unscaled)
        else:
            return sample

    def discrete_transform(self, sample: Tensor) -> Tensor:
        for variable_metadata in self.metadata.get_by_independent_variable():
            index = variable_metadata.get_feature_index()
            size = variable_metadata.get_size()
            old_value = sample[:, index:index + size]
            # discrete binary
            if variable_metadata.is_binary():
                new_value = (old_value.view(-1) > .5).view(-1, 1)
            # discrete categorical
            elif variable_metadata.is_categorical():
                new_value = one_hot(old_value.argmax(dim=1), num_classes=size)
            # leave numerical variables the untouched
            else:
                new_value = old_value
            sample[:, index:index + size] = new_value
        return sample
