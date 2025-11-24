# Machine learning for Polymer Property Predictions

Polymers are macromolecules that are composed of repeating units known as monomers. Machine learning in molecules (and particularly, polymers) has an incentive such that it can be used to screen for promising properties. However, due to the polymer's "periodicity" in its repeating units and its lack of data, it is a challenge to build a sufficient machine learning model to predict glass transition temperature (Tg), thermal conductivity (Tc), fractional free volume (FFV) and radius of gyration. This work aims to address that by building a monomer-level representation and perform classic machine learning to make property predictions.

This is for our CHE1147 course at the University of Toronto. Please refer to their GitHub repository [here](https://github.com/AI4ChemS/CHE1147).

## Data Availability
The source of the polymer labels are from the NeurIPS Open Polymer Prediction Competition from Kaggle (2025). The ```train.csv``` can be found below. The RDKit_descriptors.csv were computed by our team from ```utils/featurizer.py```. This will be further elaborated on in the Usage section.

<b>Source of data used: https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/data.</b>

## Usage
This work can be broken down into the following sections:
1. <b>Data</b>: The files ```train.csv``` and ```RDKit_descriptors.csv``` are available in the ```data``` folder. These will be referred to for the rest of the work. The ```train.csv``` come directly from the NeurIPS Open Polymer Prediction Competition, and the ```RDKit_descriptors.csv``` are composed of all RDKit descriptors and a few selected topological descriptors.
2. ```eda.ipynb``` is a notebook showcasing our full exploratory data analysis (EDA), alongside comments about it and figures.
3. ```utils/featurizer.py``` is our featurizer script, in which it contains a class ```Featurizer```, which returns RDKit descriptors and select topological descriptors. If you are interested in using this class for featurizing a SMILES string, here is an example snippet:
  ```python
from utils.featurizer import Featurizer

smi = "*CC(*)c1ccccc1C(=O)OCCCCCC"
features = Featurizer(smi).summary_of_results()
```
  
5. ```utils/optimization.py``` is the script containing functions for RFECV feature selection and hyperparameter tuning using BayesSearchCV.
6. <b>Important!</b> ```main.py``` is a script that you can run, in which it will utilize ```featurizer.py``` and ```optimization.py``` to featurize and perform RFECV + hyperparameter tuning, while returning the evaluations on the test set. These results will be saved in ```ML_results/tuned_results/label.json```. Some examples of these saved results are already there, in which the .json contains keys for: ```bscv_results```, which contains the best hyperparameters via calling the sub-dictionary ```best_parameters```, alongside the features selected from RFECV in ```features_used```. Finally, the RFECV + tuned model scores on test set are saved under key ```scores```. If you want a summary of how to run main.py:
```
cd path/to/polymer-small-data
python main.py
```
If you want to load results for some label after running main.py, here is a small Python snippet as an example:
```python
import json
import os

label = "Tc"
path_to_json = os.path.join("ML_results/tuned_results", f"{label}.json")
with open(path_to_json) as json_file:
  data = json.load(json_file)

hyperparameters = data['bscv_results']['best_parameters']
features_selected = data['features_used']
scores = data['scores'] # should be dictionary of {"SRCC" : srcc, "MAE" : mae}
```
7. 

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
