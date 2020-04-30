import time

from typing import List

import numpy as np

from sklearn.metrics import precision_recall_curve, f1_score, auc
from imblearn.pipeline import Pipeline

from augmentation.model import AugmentationTaskModelFactory
from deep_generative_models.configuration import Configuration
from deep_generative_models.tasks.multiprocess_runner import MultiProcessTaskWorker
from deep_generative_models.tasks.task import Task


class AugmentationTask(Task):
    worker: MultiProcessTaskWorker
    model_factory: AugmentationTaskModelFactory

    def __init__(self, worker: MultiProcessTaskWorker):
        super(AugmentationTask, self).__init__(logger=worker.logger)
        self.worker = worker

        self.model_factory = AugmentationTaskModelFactory()

    def mandatory_arguments(self) -> List[str]:
        return [
            "name",
            "case",
            "data",
            "model"
        ]

    @staticmethod
    def compute_pr_auc_score(model: Pipeline, features: np.ndarray, labels: np.ndarray) -> float:
        probabilities = model.predict_proba(features)[:, 1]
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        return auc(recall, precision)

    @staticmethod
    def compute_f1_score(model: Pipeline, features: np.ndarray, labels: np.ndarray) -> float:
        predictions = model.predict(features)
        return f1_score(labels, predictions)

    def run(self, configuration: Configuration) -> None:
        start_time = time.time()
        output = {}

        try:
            # validate configuration
            self.validate_arguments(configuration)

            # validate folds
            assert isinstance(configuration.data.train_features, list) \
                and isinstance(configuration.data.train_labels, list) \
                and isinstance(configuration.data.val_features, list) \
                and isinstance(configuration.data.val_labels, list) \
                and isinstance(configuration.data.test_features, list) \
                and isinstance(configuration.data.test_labels, list)

            assert configuration.data.num_folds == len(configuration.data.train_features) \
                == len(configuration.data.train_labels) \
                == len(configuration.data.val_features) \
                == len(configuration.data.val_labels) \
                == len(configuration.data.test_features) \
                == len(configuration.data.test_labels)

            # add basic information
            output["case"] = configuration.case
            output["name"] = configuration.name

            # add augmentation information
            if "under_sampling" in configuration.model:
                output["under_sampling"] = configuration.model.under_sampling.type
                output["under_sampling_ratio"] = configuration.model.under_sampling.ratio
            if "over_sampling" in configuration.model:
                output["over_sampling"] = configuration.model.over_sampling.type
                output["over_sampling_ratio"] = configuration.model.over_sampling.ratio

            # initialize
            train_pr_auc_scores = np.zeros(configuration.data.num_folds)
            test_pr_auc_scores = np.zeros(configuration.data.num_folds)
            train_f1_scores = np.zeros(configuration.data.num_folds)
            test_f1_scores = np.zeros(configuration.data.num_folds)

            # cross validation
            for fold in range(configuration.data.num_folds):
                train_features = np.concatenate((
                    np.load(configuration.data.train_features[fold]),
                    np.load(configuration.data.val_features[fold])
                ), axis=0)
                train_labels = np.concatenate((
                    np.load(configuration.data.train_labels[fold]),
                    np.load(configuration.data.val_labels[fold])
                ), axis=0)

                model = self.model_factory.create(configuration.model, fold)
                model.fit(train_features, train_labels)

                test_features = np.load(configuration.data.test_features[fold])
                test_labels = np.load(configuration.data.test_labels[fold])

                train_pr_auc_scores[fold] = self.compute_pr_auc_score(model, train_features, train_labels)
                test_pr_auc_scores[fold] = self.compute_pr_auc_score(model, test_features, test_labels)
                train_f1_scores[fold] = self.compute_f1_score(model, train_features, train_labels)
                test_f1_scores[fold] = self.compute_f1_score(model, test_features, test_labels)

            # aggregate
            output["train_pr_auc_mean"] = train_pr_auc_scores.mean()
            output["train_pr_auc_std"] = train_pr_auc_scores.std()
            output["test_pr_auc_mean"] = test_pr_auc_scores.mean()
            output["test_pr_auc_std"] = test_pr_auc_scores.std()
            output["train_f1_mean"] = train_f1_scores.mean()
            output["train_f1_std"] = train_f1_scores.std()
            output["test_f1_mean"] = test_f1_scores.mean()
            output["test_f1_std"] = test_f1_scores.std()

            # there were no errors
            output["has_error"] = False

        # there was an error
        except Exception as e:
            output["has_error"] = True
            output["error"] = str(e)

        # add time
        output["time"] = time.time() - start_time

        # send output
        self.worker.send_output(output)
