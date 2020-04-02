import numpy as np

from deep_generative_models.architecture import Architecture
from deep_generative_models.checkpoints import Checkpoints
from deep_generative_models.configuration import Configuration, load_configuration
from deep_generative_models.factories import create_architecture
from deep_generative_models.gpu import to_cpu_if_was_in_gpu
from deep_generative_models.metadata import load_metadata, Metadata
from deep_generative_models.tasks.task import Task

from torch import Tensor, no_grad


class Sample(Task):

    def run(self, configuration: Configuration) -> None:
        metadata = load_metadata(configuration.metadata)

        architecture = create_architecture(metadata, load_configuration(configuration.architecture))
        architecture.to_gpu_if_available()

        checkpoints = Checkpoints(configuration.checkpoint)
        checkpoint = checkpoints.load()
        checkpoints.load_states(checkpoint["architecture"], architecture)

        samples = np.zeros((configuration.num_samples, metadata.get_num_features()), dtype=np.float32)

        start = 0
        end = 0
        iterations = 0
        while start < configuration.num_samples:
            # do not calculate gradients
            with no_grad():
                # sample from the model
                batch_samples = self.generate_sample(configuration, metadata, architecture)

            # transform back the samples
            batch_samples = to_cpu_if_was_in_gpu(batch_samples)
            batch_samples = batch_samples.numpy()

            # recalculate after filtering
            real_batch_size = len(batch_samples)

            # if there was a change
            if real_batch_size > 0:
                # do not go further than the desired number of samples
                end = min(start + real_batch_size, configuration.num_samples)
                # limit the samples taken from the batch based on what is missing
                samples[start:end, :] = batch_samples[:min(real_batch_size, end - start), :]

                # restart iterations
                iterations = 0

                # move to next batch
                start = end

            # if there was no change count another iteration
            else:
                # try again
                iterations += 1

            # stop in case there is an iteration limit
            if "max_iterations" in configuration and iterations >= configuration.max_iterations:
                raise Exception("Reached maximum number of iterations with sample size {:d}.".format(end))

        # save the samples
        np.save(configuration.output, samples)

    def generate_sample(self, configuration: Configuration, metadata: Metadata, architecture: Architecture) -> Tensor:
        raise NotImplementedError
