# Heart Disease Risk Analysis

## Overview

This repository contains the analysis conducted for Individual Task 1 in Data Science. The project investigates heart disease risk using two publicly available datasets collected in different settings.

The analysis compares two machine learning approaches:

- Logistic Regression
- Random Forest

The models are evaluated using healthcare-relevant metrics, with particular emphasis on recall because failing to identify a person with heart disease is more consequential than generating a false positive.

## Datasets

Two datasets are used:

1. **Cleveland Heart Disease Dataset**  
   A clinical dataset containing patient-level medical measurements and heart disease outcomes.

2. **BRFSS Heart Disease Dataset**  
   A large public-health survey dataset containing self-reported health, lifestyle and demographic indicators.

The datasets differ substantially in size and data collection methodology, allowing the analysis to examine whether similar predictive patterns emerge across different sources.

## Analysis

The project includes:

- Data preparation and cleaning
- Exploratory analysis and correlation analysis
- Logistic Regression
- Random Forest
- Class-imbalance handling
- Model evaluation
- Confusion matrices
- Feature-importance analysis
- Cross-dataset comparison

The analysis focuses on identifying predictive patterns associated with heart disease and comparing the relative importance of blood-pressure and diabetes-related indicators with other health factors.

## Repository Structure

```text
data/
    heart_disease_brfss.csv
    heart_disease_cleveland.csv

src/
    heart_disease_brfss_analysis.py
    heart_disease_cleveland_analysis.py
    compare_datasets.py
    model_utils.py

outputs/
    Model results, graphs, feature-importance files
    and evaluation outputs

requirements.txt
