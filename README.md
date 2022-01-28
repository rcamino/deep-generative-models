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

Also make sure that the project directory root is in the PATH to execute the rest of the scripts.

## Data

This library works with data in numpy format.
Categorical variables should be one-hot-encoded and numerical variables should be scaled between 0 and 1.
You can find example scripts to pre-process data this way in [rcamino/dataset-pre-processing](https://github.com/rcamino/dataset-pre-processing).

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

You can find example scripts to generate this kind of metadata in [rcamino/dataset-pre-processing](https://github.com/rcamino/dataset-pre-processing).

## Runner

The runner is the main entry point for this library. It receives a configuration file as the only argument.

```bash
python deep_generative_models/tasks/runner.py configuration.json
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

- `"SingleVariableAutoEncoder"`
- `"MultiVariableAutoEncoder"`
- `"SingleInputEncoder"`
- `"MultiInputEncoder"`
- `"SingleOutputDecoder"`
- `"MultiOutputDecoder"`
- `"SingleOutputGenerator"`
- `"MultiOutputGenerator"`
- `"Discriminator"`
- `"Critic"`
- `"SingleVariableDeNoisingAutoencoder"`
- `"MultiVariableDeNoisingAutoencoder"`
- `"SingleVariableVAE"`
- `"MultiVariableVAE"`
- `"CodeGenerator"`
- `"CodeDiscriminator"`
- `"CodeCritic"`
- `"SingleVariableGAINGenerator"`
- `"MultiVariableGAINGenerator"`
- `"SingleInputGAINDiscriminator"`
- `"MultiInputGAINDiscriminator"`
- `"SingleVariableMIDA"`
- `"MultiVariableMIDA"`

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