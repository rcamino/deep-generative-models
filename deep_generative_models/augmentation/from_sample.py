from typing import Tuple

import numpy as np


class OverSamplerFromSample:
    minority_sample: np.ndarray
    ratio: float
    minority_class: int

    def __init__(self, minority_sample: np.ndarray, ratio: float, minority_class: int = 1) -> None:
        self.minority_sample = minority_sample
        self.ratio = ratio
        self.minority_class = minority_class

    def fit_resample(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert len(features) == len(labels)

        # calculate sample sizes
        old_num_minority_samples = int(np.sum(labels == self.minority_class))
        num_majority_samples = len(features) - old_num_minority_samples
        new_num_minority_samples = int(num_majority_samples * self.ratio)

        if new_num_minority_samples < old_num_minority_samples:
            raise Exception("Should get a bigger sample after over sampling.")

        needed_num_minority_samples = new_num_minority_samples - old_num_minority_samples

        if needed_num_minority_samples > self.minority_sample.shape[0]:
            raise Exception("Sample is too small.")

        # sub-sample the sample
        indices = np.arange(self.minority_sample.shape[0])
        np.random.shuffle(indices)
        indices = indices[:needed_num_minority_samples]

        # augment the features
        sample_features = self.minority_sample[indices, :]
        over_sampled_features = np.concatenate((features, sample_features))

        # augment the labels
        sample_labels = np.ones(needed_num_minority_samples, dtype=np.int32) * self.minority_class
        over_sampled_labels = np.concatenate((labels, sample_labels))

        # shuffle the augmented data
        indices = np.arange(over_sampled_features.shape[0])
        np.random.shuffle(indices)
        return over_sampled_features[indices, :], over_sampled_labels[indices]
