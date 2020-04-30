from typing import Tuple

import numpy as np

from imblearn.over_sampling import SMOTENC

from deep_generative_models.metadata import Metadata


class WrappedSMOTENC:
    metadata: Metadata
    ratio: float

    def __init__(self, metadata: Metadata, ratio: float) -> None:
        self.metadata = metadata
        self.ratio = ratio

    def fit_resample(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        assert len(features) == len(labels)

        # flags indicating which variables are label encoded SMOTE-NC
        is_le = []
        # start index of the current variable
        ohe_index = 0
        # there will be one label encoded feature per variable
        # so we can use the variable index as the label encoded feature index
        le_features = np.zeros((len(features), self.metadata.get_num_independent_variables()),
                               dtype=features.dtype)
        for variable_metadata in self.metadata.get_by_independent_variable():
            le_index = variable_metadata.get_index()
            ohe_size = variable_metadata.get_size()

            # categorical
            if variable_metadata.is_categorical():
                variable_features = features[:, ohe_index:ohe_index + ohe_size]
                assert variable_features.shape[1] == ohe_size
                le_features[:, le_index] = np.argmax(variable_features, axis=1)
                ohe_index += ohe_size
                is_le.append(True)
            # numerical or binary
            else:
                # for both cases
                assert ohe_size == 1
                le_features[:, le_index] = features[:, ohe_index]
                ohe_index += 1
                # it is considered categorical if it is binary
                # it is not considered categorical if it is numerical
                is_le.append(variable_metadata.is_binary())

        # over sample
        wrapped = SMOTENC(is_le, sampling_strategy=self.ratio)
        le_os_features, os_labels = wrapped.fit_resample(le_features, labels)
        os_size = len(os_labels)

        # now we need to one hot encode again
        ohe_os_features = np.zeros((os_size, self.metadata.get_num_features()), dtype=features.dtype)
        ohe_index = 0
        for variable_metadata in self.metadata.get_by_independent_variable():
            le_index = variable_metadata.get_index()
            ohe_size = variable_metadata.get_size()

            # categorical
            if variable_metadata.is_categorical():
                le_value = le_os_features[:, le_index].astype(np.int)
                ohe_os_features[:, ohe_index + le_value] = 1
                ohe_index += ohe_size
            # numerical or binary
            else:
                # for both cases
                assert ohe_size == 1
                ohe_os_features[:, ohe_index] = le_os_features[:, le_index]
                ohe_index += 1

        # return the one hot encoded sample
        return ohe_os_features, os_labels
