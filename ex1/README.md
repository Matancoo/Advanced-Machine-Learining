## Project Overview

This project `ex1` is a collection of models and utilities for AutoRegressive and Diffusion models. The repository is organized into two main directories, each focusing on a distinct type of model. 

## Directory Structure

Here's a high-level overview of our directory structure:

ex1/
-- AutoRegressive/
   |-- model/
   |-- minGPT/
   |-- bpe.py
   |-- ar_tasks.py
   |-- example_gpt.py
   |-- alice_in_wonderland.txt
-- Diffusion/
   |-- model/
   |-- conditional_diffusion_model.py
   |-- diffusion_model.py
   |-- example_diffusion.py
   |-- helpers_and_metrics.py
   |-- unconditional_diffusion_model.py

### AutoRegressive

The `AutoRegressive` folder contains everything needed to train and run autoregressive models. The notable files and folders in this directory are:

- `model/`: This directory contains saved models that have been trained.
- `minGPT/`: This directory contains the `minGPT` model implementation. 
  - `bpe.py`: Byte Pair Encoding (BPE) utilities.
  - `model.py`: The Transformer model implementation.
  - `trainer.py`: Training utilities for the model.
  - `utils.py`: Other various utilities used by the model.
- `bpe.py`: Top-level Byte Pair Encoding (BPE) utility.
- `ar_tasks.py`: Defines autoregressive tasks for the model to perform.
- `example_gpt.py`: Example script showing how to train and run the GPT model.
- `alice_in_wonderland.txt`: An example text file used for training.

### Diffusion

The `Diffusion` folder contains everything needed to train and run diffusion models. The notable files and folders in this directory are:

- `model/`: This directory contains saved models that have been trained.
- `conditional_diffusion_model.py`: Defines the conditional diffusion model.
- `diffusion_model.py`: Defines the general structure of diffusion models.
- `example_diffusion.py`: Example script showing how to train and run diffusion models.
- `helpers_and_metrics.py`: Contains helper functions and metric calculations for diffusion models.
- `unconditional_diffusion_model.py`: Defines the unconditional diffusion model.
