from typing import List, Type, Any

import numpy as np
import xgboost as xgb

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, KMeansSMOTE, RandomOverSampler, SMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler

from augmentation.from_sample import OverSamplerFromSample
from augmentation.wrapped_smote_nc import WrappedSMOTENC
from deep_generative_models.arguments import ArgumentValidator
from deep_generative_models.configuration import Configuration
from deep_generative_models.metadata import load_metadata


class ImbalancedLearnFactory:

    def create(self, metadata_path: str, ratio: float) -> Any:
        raise NotImplementedError


class ImbalancedLearnFactoryFromClass(ImbalancedLearnFactory):
    imbalanced_learn_class: Type

    def __init__(self, imbalanced_learn_class: Type) -> None:
        self.imbalanced_learn_class = imbalanced_learn_class

    def create(self, metadata_path: str, ratio: float) -> Any:
        return self.imbalanced_learn_class(sampling_strategy=ratio)


class SMOTENCFactory(ImbalancedLearnFactory):

    def create(self, metadata_path: str, ratio: float) -> Any:
        return WrappedSMOTENC(load_metadata(metadata_path), ratio)


IMBALANCED_LEARN_FACTORY_BY_NAME = {
    "ADASYN": ImbalancedLearnFactoryFromClass(ADASYN),
    "BorderlineSMOTE": ImbalancedLearnFactoryFromClass(BorderlineSMOTE),
    "KMeansSMOTE": ImbalancedLearnFactoryFromClass(KMeansSMOTE),
    "RandomOverSampler": ImbalancedLearnFactoryFromClass(RandomOverSampler),
    "SMOTE": ImbalancedLearnFactoryFromClass(SMOTE),
    "SMOTENC": SMOTENCFactory(),
    "SVMSMOTE": ImbalancedLearnFactoryFromClass(SVMSMOTE),
}


class AugmentationTaskModelFactory(ArgumentValidator):

    def mandatory_arguments(self) -> List[str]:
        return [
            "n_estimators",
            "max_depth"
        ]

    def optional_arguments(self) -> List[str]:
        return [
            "under_sampling",
            "over_sampling"
        ]

    def create(self, configuration: Configuration, fold: int) -> Pipeline:
        self.validate_arguments(configuration)

        steps = []

        # under sampling
        if "under_sampling" in configuration:
            if "type" not in configuration.under_sampling:
                raise Exception("Undefined under-sampling type.")

            # random under sampling
            if configuration.under_sampling.type == "random":
                under_sampler = RandomUnderSampler(sampling_strategy=configuration.under_sampling.ratio)

            # unexpected under sampling type
            else:
                raise Exception("Unexpected under-sampling type '{}'.".format(configuration.under_sampling.type))

            steps.append(("under-sampling", under_sampler))

        # over sampling
        if "over_sampling" in configuration:
            if "type" not in configuration.over_sampling:
                raise Exception("Undefined over-sampling type.")

            # over sampling with imbalanced learn
            if configuration.over_sampling.type == "imbalanced_learn":
                imbalanced_learn_factory = IMBALANCED_LEARN_FACTORY_BY_NAME[configuration.over_sampling.imbalanced_learn_factory]

                over_sampler = imbalanced_learn_factory.create(configuration.over_sampling.metadata_path,
                                                               configuration.over_sampling.ratio)

            # over sampling from sample
            elif configuration.over_sampling.type == "from_sample":
                over_sampler = OverSamplerFromSample(np.load(configuration.over_sampling.sample_path_by_fold[fold]),
                                                     configuration.over_sampling.ratio)
            # unexpected over sampling type
            else:
                raise Exception("Unexpected over-sampling type '{}'.".format(configuration.over_sampling.type))

            steps.append(("over-sampling", over_sampler))

        # classifier
        classifier = xgb.XGBClassifier(n_estimators=configuration.n_estimators,
                                       max_depth=configuration.max_depth,
                                       n_jobs=1)

        steps.append(("classifier", classifier))

        return Pipeline(steps)
