# Collaborative Expert Learning with Clique-aware Graph Propagation for Aesthetic Perception Modeling

This repository provides a PyTorch implementation of the Collaborative Expert Learning with Clique-aware Graph Propagation for Aesthetic Perception Modeling presented in our KDD 2026 paper.

## Get Started

### Requirements and Installation

The require environments are in **environment.yaml**, you can run below command to install the environment:

```python
conda env create -f environment.yaml
```

Optionally, you can install the environment through the **requirements.txt** as below:
```python
pip install -r requirements.txt 
```


## Usage

First, create a result directory to store the results(if no results directory exists). Then, train the model by running main.py directly. Remember to set the chosen dataset, model backbone and hyper-parameters in the script.Please use the following command to load demo data for testing whether the environment is successfully installed.

```python
python main.py --data_path ./data --res_save_path result
```

optional arguments:  
```--help``` show this help message and exit  
```--data_path``` show the the path of data  
```--batchSize``` show the size of a single batch of data 
## Note

The data used in this experiment is anonymous and temporarily not publicly available, so encrypted data (data.json, reflect.json, dimension_info.json) is used.
