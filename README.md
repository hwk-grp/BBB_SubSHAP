# Explaining Blood-Brain Barrier Permeability by Synergistic Effect on Molecular Substructures

This repository focuses on predicting and explaining Blood-Brain Barrier (BBB) Permeability using explainable machine learning (ML) methods. The core idea is to identify the synergistic effects of molecular substructures contributing to BBB permeability and other related mechanisms.

## Repository Structure

- **data/**: Contains dataset files used for model training and evaluation.
- **preds/**: Stores model prediction outputs.
- **utils/**: Utility functions and scripts used across the project.
- **bbbp_env.yml**: Conda environment file with all necessary dependencies to run the project.

## Installation

1. Download the files

2. Create the conda environment:
conda env create -f bbbp_env.yml

## Usage

Run the desired models or analysis using the provided Python scripts:

python run_DNN.py            # Run Deep Neural Network
python run_GNN.py            # Run Graph Neural Network
python run_XAI_model.py      # Save trained XAI model 
python SHAP_analysis.py      # Interpret predictions with SHAP 
python synergy_group_analysis.py      # Interpret synergistic effect between substructures  
