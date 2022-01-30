# Deep Generative Models

The goal of this library is to automatize several tasks related to Deep Generative Models based on json configuration files.

## Requirements

This code was developed with Python 3.
All the python libraries required to run this code are listed on the file `requirements.txt`.
I recommend installing everything with pip inside a virtual environment.
Run this code inside the project directory root:

```bash
pip install -r requirements.txt
```

Make sure that the project main module (`deep_generative_models`) can be found by python.
There are several ways to achieve this:

- By executing the scripts from the project root directory.
- Adding the project root directory to the `PYTHONPATH` environment variable (see python [documentation](https://docs.python.org/3/using/cmdline.html#envvar-PYTHONPATH)).
- Adding a `.pth` file to the site-packages directory of the virtual environment with the absolute path of the root directory (see python [documentation](https://docs.python.org/3/library/site.html)).

## Data

This library works with data in numpy format.
Categorical variables should be one-hot-encoded and numerical variables should be scaled between 0 and 1.

## Metadata

Some specific information about the data is needed for this library to work.
Metadata json files should have the following format:

```json
{
  "variables": [],
  "variable_sizes": [],
  "variable_types": [],
  "features": [],
  "index_to_value": [],
  "value_to_index": {},
  "num_features": 0,
  "num_variables": 0,
  "num_binary_variables": 0,
  "num_categorical_variables": 0,
  "num_numerical_variables": 0,
  "num_samples": 0,
  "num_classes": 0,
  "classes": []
}
```
Properties:

- `"variables"` (string array): Each position determines the name of the corresponding variable.
- `"variable_types"` (string array): Each position determines the type of the corresponding variable and can take the value `"categorical"`, `"numerical"`, or `"binary"`.
- `"variable_sizes"` (int array): Each position determines the size of the corresponding variable (see constraints).
- `"num_variables"` (int): Number of variables.
- `"num_binary_variables"` (int): Number of variables with binary type.
- `"num_categorical_variables"` (int): Number of variables with categorical type.
- `"num_numerical_variables"` (int): Number of variables with numerical type.
- `"features"` (string array): Each position determines the name of the corresponding feature.
- `"num_features"` (int): Number of features.
- `"value_to_index"` (dictionary of dictionaries): The mapping `i = value_to_index["category"]["value"]` indicates that the variable with name `"category"` is mapped to the feature with index `i` when the variable value is equal to `"value"`.
- `"index_to_value"` (string-array array): Each position `i` of `index_to_value` has the form `["category", "value"]`, which is the inverse of `value_to_index` (see constraints).
- `"num_samples"` (int): Number of samples.
- `"num_classes"` (optional int): Number of different classes (if present).
- `"classes"` (optional string array): The name of each class (if present).

I am aware that some values are redundant, but if they are pre-calculated I can save some time.

Constraints:

- `len(variables) == len(variable_sizes) == len(variable_types) == num_variables > 0`
- `sum([1 for i in range(num_variables) if variable_types[i] == "binary"]) == num_binary_variables >= 0`
- `sum([1 for i in range(num_variables) if variable_types[i] == "categorical"]) == num_categorical_variables  >= 0`
- `sum([1 for i in range(num_variables) if variable_types[i] == "numerical"]) == num_numerical_variables >= 0`
- `num_variables = num_binary_variables + num_categorical_variables + num_numerical_variables`
- ``variable_sizes[i] > 2 for i in range(num_variables) if variable_types[i] == "categorical"``
- ``variable_sizes[i] == 1 for i in range(num_variables) if variable_types[i] != "categorical"``
- `len(features) == len(index_to_value) == num_features >= num_variables`
- ```
  index_to_value[index][0] == category and index_to_value[index][1] == value
  for category in value_to_index.keys()
  for value, index in value_to_index[category].items()
  ```

## Runner

The runner is the main entry point for this library. It receives a configuration file as the only argument.

```bash
python -m deep_generative_models.tasks.runner.py configuration.json
```

The file `configuration.json` should have the following format:

```json
{
  "task": "...",
  "arguments": {}
}
```

The `"task"` can take any of the following string values:

- Train:
  - `"TrainARAE"`
  - `"TrainAutoEncoder"`
  - `"TrainDeNoisingAutoEncoder"`
  - `"TrainGAIN"`
  - `"TrainGAN"`
  - `"TrainMIDA"`
  - `"TrainMedGAN"`
  - `"TrainVAE"`
- Sample:
  - `"SampleARAE"`
  - `"SampleGAN"`
  - `"SampleMedGAN"`
  - `"SampleVAE"`
- Encode:
  - `"Encode"`
- Imputation:
  - `"ComputeMeansAndModes"`
  - `"GANIterativeImputation"`
  - `"GenerateMissingMask"`
  - `"ImputeWithGAIN"`
  - `"ImputeWithMIDA"`
  - `"ImputeWithVAE"`
  - `"MeansAndModesImputation"`
  - `"MissForestImputation"`
  - `"NormalNoiseImputation"`
  - `"TrainMissForest"`
  - `"ZeroImputation"`
- Tasks that run other tasks:
  - `"MultiProcessTaskRunner"`
  - `"SerialTaskRunner"`

The arguments depend on the task (see each task documentation).

## Tasks

The task type defines the list of mandatory and optional arguments they possess.
Some tasks are a "base" for other tasks and cannot be executed.
The runnable tasks inherit the arguments for their base tasks, and can be executed in two ways:

- using the [runner](#runner),
- using the task script directly (see each task documentation).

### Base Tasks

#### Impute
#### Sample
#### Train

### Runnable Tasks

#### Compute Means And Modes
#### Encode
#### Generate Missing Mask
#### Impute With MIDA
#### Means And Modes Imputation
#### MissForest Imputation
#### Multi-Process Task Runner
#### Normal Noise Imputation
#### Sample ARAE
#### Sample GAN
#### Sample MedGAN
#### Sample VAE
#### Serial Task Runner
#### Train ARAE
#### Train AutoEncoder
#### Train DeNoising AutoEncoder
#### Train GAIN
#### Train GAN
#### Train MIDA
#### Train MedGAN
#### Train MissForest
#### Train VAE
#### Zero Imputation

## Architecture

The architecture json file should have the following format:

```json
{
  "arguments": {},
  "components": {}
}
```

The `"arguments"` object contains global architecture configuration which is shared among components.
The key for each element in the `"coponents"` object defines a reference name for the component, and the value should have the following format:

```json
{
  "factory": "...",
  "arguments": {}
}
```

The `"factory"` defines who will create the component and can take any of the following string values:

- `"CodeCritic"`
- `"CodeDiscriminator"`
- `"CodeGenerator"`
- `"Critic"`
- `"Discriminator"`
- `"MultiInputEncoder"`
- `"MultiInputGAINDiscriminator"`
- `"MultiOutputDecoder"`
- `"MultiOutputGenerator"`
- `"MultiVariableAutoEncoder"`
- `"MultiVariableDeNoisingAutoencoder"`
- `"MultiVariableGAINGenerator"`
- `"MultiVariableMIDA"`
- `"MultiVariableVAE"`
- `"SingleInputEncoder"`
- `"SingleInputGAINDiscriminator"`
- `"SingleOutputDecoder"`
- `"SingleOutputGenerator"`
- `"SingleVariableAutoEncoder"`
- `"SingleVariableDeNoisingAutoencoder"`
- `"SingleVariableGAINGenerator"`
- `"SingleVariableMIDA"`
- `"SingleVariableVAE"`

Components at the same time can reference to other factories:

- Layers:
  - `"AdditiveNormalNoise"`
  - `"HiddenLayers"`
  - `"MeanAndModesImputation"`
  - `"MultiInputDropout"`
  - `"MultiInputLayer"`
  - `"MultiOutputLayer"`
  - `"NormalNoiseImputation"`
  - `"SingleInputLayer"`
  - `"SingleOutputLayer"`
  - `"ZeroImputation"`
- Activations:
  - `"GumbelSoftmaxSampling"`
  - `"SoftmaxSampling"`
- Losses:
  - `"AutoEncoderLoss"`
  - `"GAINDiscriminatorLoss"`
  - `"GAINGeneratorLoss"`
  - `"GANDiscriminatorLoss"`
  - `"GANGeneratorLoss"`
  - `"MultiReconstructionLoss"`
  - `"RMSE"`
  - `"VAELoss"`
  - `"ValidationImputationLoss"`
  - `"ValidationReconstructionLoss"`
  - `"WGANCriticLoss"`
  - `"WGANCriticLossWithGradientPenalty"`
  - `"WGANGeneratorLoss"`
- Optimizers:
  - `"WGANOptimizer"`

There are also factories for PyTorch elements (see the argument definition in the corresponding PyTorch documentation):

- Layers:
  - `"Dropout"`
- Activations:
  - `"LeakyReLU"`
  - `"ReLU"`
  - `"Sigmoid"`
  - `"Tanh"`
- Losses:
  - `"BCE"`
  - `"CrossEntropy"`
  - `"MSE"`
- Optimizers:
  - `"Adam"`
  - `"SGD"`

## Examples

*IMPORTANT:* The following examples do not describe by any means the best combination of architecture or hyper-parameters to achieve good results.
The goal of the examples is to illustrate how to use this library.

### Data and Metadata

Data is not included in the examples because of space limitations and because it does not belong to me.
You can find example scripts to download the kind of data and metadata required for this project in [rcamino/dataset-pre-processing](https://github.com/rcamino/dataset-pre-processing).

For example, to download and pre-process the "Default of Credit Card Clients" dataset from the UCI repository, first create the directories `examples/data/default_of_credit_card_clients` and then execute:

```bash
python $DATASET_PRE_PROCESSING_ROOT/uci/default_of_credit_card_clients/download_and_transform.py \
  --directory=examples/data/default_of_credit_card_clients
```

If you want to run examples that need the data split into train and test (i.e. 90 train / 10% test split):

```bash
python $DATASET_PRE_PROCESSING_ROOT/train_test_split.py --stratify --shuffle \
  $DIRECTORY\features.npy \
  0.9 \
  $DIRECTORY\features-train.npy \
  $DIRECTORY\features-test.npy \
  --labels=$DIRECTORY\labels.npy \
  --train_labels=$DIRECTORY\labels-train.npy \
  --test_labels=$DIRECTORY\labels-test.npy
```

You can find more information about the dataset in the [dataset repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) or in [my code repository](https://github.com/rcamino/dataset-pre-processing/tree/master/dataset_pre_processing/uci/default_of_credit_card_clients).

### Tasks

For simplicity, these tasks use paths relative to the project root.
This means that unless you want to edit the tasks yourself, most of these tasks will expect that:

- data can be found in `examples/data`,
- and other tasks configuration files can be found in `examples/tasks`.

The tasks and architectures depend heavily on the data and metadata. 

### Train and sample from a single-variable GAN using the Default of Credit Card Clients dataset

- Task directory: `examples/tasks/default_of_credit_card_clients/single_variable_gan`.
- Expected data directory: `examples/data/default_of_credit_card_clients`.
- Architecture (`architecture.json`):
  - It is called "single-variable" (in opposed to "multi-variable") because it does not take into account the types of each variable.
  The metadata is basically ignored.
  It would be the more "traditional" way of training GANs.
  - The generator and the discriminator have only one hidden layer of size 50.
  - The input noise for the generator is also of size 50.
  - The generator and discriminator training steps use an Adam optimizer with learning rate 0.001.
- Tasks:
  - `train.json`: the train task. 
  - `sample.json`: the sample task (depends on the results of the train task).
  - `train-then-sample.json`: it is just a serial task that runs the train and then the sample task.
- Files that will be generated:
  - `checkpoint.pickle`: the trained models.
  - `train.csv`: training information (i.e. loss).
  - `sample.npy`: a small dataset generated by the trained model.