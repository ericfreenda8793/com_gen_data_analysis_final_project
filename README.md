# Tomato Genome-Wide Association Studies (GWAS) Repository

## Overview
This repository contains the datasets, Python notebooks, and evaluation metrics used for conducting Genome-Wide Association Studies (GWAS) on tomato traits. It includes a variety of machine learning models and visualizations of the features detected by each model.

## Repository Structure
- Run each individual model in the current directory
- `/Features_selected`: Directory where features selected by each model are exported.
- `/feature_evaluation.ipynb`: Notebook for evaluating the performance of the models based on the selected features.

## Data
The repository utilizes a tomato dataset specifically prepared for GWAS testing. This dataset includes phenotypic and genotypic information essential for the studies.

## Running the Models
To run the individual models and export the selected features:
1. Navigate to the `/models` directory.
2. Open each notebook corresponding to a model.
3. Execute the notebook cells to perform the analysis and export the selected features to the `/Features_selected` directory.

## Evaluating Model Performance
To evaluate the performance of each model based on the features it selected:
1. Open the `feature_evaluation.ipynb` notebook located in the root directory.
2. Run the notebook to see the evaluation metrics and performance comparisons of the models.

## Visualization
- The feature visualization graph in the repository displays all features detected across all models, providing a comprehensive overview of the feature selection consistency.

## Contributions
Contributions to this repository are welcome. Please ensure to update tests as appropriate and adhere to the existing coding style.

## License
Specify the license under which your project is made available. (e.g., MIT, GPL-3.0, etc.)
