# Machine learning for Polymer Property Predictions

Polymers are macromolecules that are composed of repeating units known as monomers. Machine learning in molecules (and particularly, polymers) has an incentive such that it can be used to screen for promising properties. However, due to the polymer's "periodicity" in its repeating units and its lack of data, it is a challenge to build a sufficient machine learning model to predict glass transition temperature (Tg), thermal conductivity (Tc), fractional free volume (FFV) and radius of gyration. This work aims to address that by building a monomer-level representation and perform classic machine learning to make property predictions.

This is for our CHE1147 course at the University of Toronto. Please refer to their GitHub repository [here](https://github.com/AI4ChemS/CHE1147).

## Data Availability
The source of the polymer labels are from the NeurIPS Open Polymer Prediction Competition from Kaggle (2025). The ```train.csv``` can be found below. The RDKit_descriptors.csv were computed by our team from ```utils/featurizer.py```. This will be further elaborated on in the Usage section.

<b>Source of data used: https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/data.</b>

## Usage


## OS Requirements
This package should be working properly on Linux and Windows. In particular, the package has been tested on Ubuntu 22.04.4 LTS.

## Installation
Python 3.11.9 is recommended for this package. No GPU is required for all the feature selection and hyperparameter tuning, although it is highly encouraged that a strong CPU is required. For a full installation of this package, please refer to the instructions below:

```
git clone https://github.com/sartaajkhan/polymer-small-data.git
conda create -n polymer-small python=3.11.9
conda activate polymer-small

cd path/to/polymer-small-data
pip install -r requirements.txt
```

Under the assumption that this is being installed on a fresh environment, the installation time should be around 2 minutes.
