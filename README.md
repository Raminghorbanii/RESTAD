# RESTAD: Reconstruction and Similarity Transformer for time series Anomaly Detection
 
This repository contains the code and additional resources used in our study. Below are links to more detailed information on the datasets used and the hyperparameters settings.

- [Read the full paper here]()

---

## Detailed Information

- [Dataset Details](Datasets_info.md): Comprehensive information on the datasets used in our experiments.
- [Hyperparameter Settings](Hyperparameters_info.md): Detailed descriptions of the hyperparameters used for our model.

---

 
## Installation Instructions

### Set Up the Environment
To reproduce the experiments and use RESTAD, create and activate a new Conda environment:

```bash
conda create --name restad_env python=3.8
conda activate restad_env
```

### Install Dependencies
Within the environment, move to this repository directory and install the required Python packages:

```bash
cd .../this_repo/restad
pip install -r requirements.txt
```

### restad directory Structure
The codebase includes scripts for setting up the model, loading configurations, training, and evaluation.

- `configs/`: Configuration files for datasets and models.
  - `dataset/`: Specific configurations for datasets.
  - `model/`: Model configurations.
- `Transformer_Model.py`: Transformer model implementation.
- `RBF_Layer.py`: Implementation of the RBF layer used within the Transformer model.
- `Utils.py`: Utility functions for data handling and other common tasks.
- `solver.py`: Contains routines for solving optimization problems.
- `evaluation.py`: Scripts to evaluate the model performance.
- `Training.py`: Core training routines for the model.
- `stages_training.py`: Script for stage-wise training.
- `main.py`: Main executable script to run experiments.


## Usage

### System Requirements
Ensure you have the appropriate computational resources available to run the project. The code was developed and tested on an **NVIDIA GeForce RTX 2080 Ti** GPU. To achieve similar performance and efficiency, it is recommended to use a comparable setup.

### Data Preparation
Download the datasets from the provided [Datasets Details](Datasets_info.md) and place them into the `restad/datasets/` directory. Ensure to update the paths in the `configs/dataset/` configuration files accordingly.

### Configuration
By default, the system uses the `base_config.yaml` file located in the `configs` directory. Modify this file to select the dataset and the initialization strategy for the RESTAD model. Hyperparameters and model configurations can be adjusted in the corresponding YAML files within the `configs/model/` directory.

### Running the Project
To run a provided trained model without undergoing the training process, if you do not have access to the required system specifications or prefer to use pre-trained models for reproducibility, you can load and run a provided trained model, placed at `restad/trained_models/` directory, by using the following command:

```bash
python main.py --load_model True
``` 

This command will skip the training process and use the pre-trained model configurations specified in your setup. To train the model from scratch, simply run:


---
## Citation
If you find our work is useful in your research, please consider raising a star  :star:  and citing:

```
@article{ghorbani2024restad,
  title={RESTAD: REconstruction and Similarity based Transformer for time series Anomaly Detection},
  author={Ghorbani, Ramin and Reinders, Marcel JT and Tax, David MJ},
  journal={arXiv preprint arXiv:2405.07509},
  year={2024}
}
```
python main.py 
``` 

