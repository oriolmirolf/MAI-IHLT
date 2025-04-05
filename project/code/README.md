# Semantic Textual Similarity Project (SemEval 2012 Task 6)

## Overview

This repository implements a system for Semantic Textual Similarity (STS) as part of the SemEval 2012 Task 6. The project focuses on detecting and quantifying the semantic similarity between pairs of sentences using various feature extraction methods and models. This work is part of the **Introduction to Human Language Technologies** (IHLT) course ([Project Description](https://smedina-upc.github.io/ihlt/sts/index.html#1
))

This project was conducted by:
- Oriol Miró
- Niklas Long

Our method is based on the UKP ([UKP Team Report](https://smedina-upc.github.io/ihlt/sts/docs/1-ukp.pdf)) and TakeLab ([TakeLab Team Report](https://smedina-upc.github.io/ihlt/sts/docs/2-takelab.pdf)) teams, which participated in SemEval 2012.

## Task Description

Semantic Textual Similarity (STS) evaluates the degree to which two sentences express the same meaning. Given a pair of sentences, a similarity score from 0 to 5 is assigned:
- **0**: Completely unrelated
- **5**: Fully equivalent

The project uses datasets provided by SemEval 2012. No external data or pre-trained embeddings are allowed.

The evaluation metric is **Pearson correlation**, comparing predicted scores with the gold standard.

## How to Reproduce

1. **Install Dependencies**

   Use the following command to install all required libraries:

   ```bash
   pip install -r requirements.txt
   ```

2. **Train the ESA Model (Optional)**

   To train the ESA (Explicit Semantic Analysis) model, you must prepare the data using the instructions provided in the [Wikipedia 2 Corpus guide](https://github.com/GermanT5/wikipedia2corpus?tab=readme-ov-file#wikipedia-2-corpus). 

   We have provided a pre-trained ESA model in [`models/esa/`](./models/esa/), trained on 300,000 Wikipedia articles. If you want to improve results, you may train it on a larger corpus.

3. **Run Experiments**

   Use [`Experiments.ipynb`](./Experiments.ipynb) to:
   - Load the data
   - Extract features (lexical, syntactic, semantic, stylistic)
   - Train models using these features

4. **Analyze Results**

   Use [`Results_and_Analysis.ipynb`](./Results_and_Analysis.ipynb) to:
   - Evaluate model performance
   - Compare Pearson correlation scores for different approaches
   - Visualize the analysis and insights

## Project Structure

- [`data/`](./data/) contains the datasets used for training and testing.
  - [`train/`](./data/train/) contains the training dataset.
  - [`test-gold/`](./data/test-gold/) contains the test dataset with gold-standard annotations.
- [`models/`](./models/) contains trained models and intermediate files.
  - [`esa/`](./models/esa/) contains ESA (Explicit Semantic Analysis) files.
    - [`esa_index.npz`](./models/esa/esa_index.npz) stores the ESA index for semantic analysis.
    - [`esa_terms.pkl`](./models/esa/esa_terms.pkl) stores the ESA term mapping.
  - [`best_model_combined.joblib`](./models/best_model_combined.joblib) stores the best model using combined features.
  - [`best_model_lexical.joblib`](./models/best_model_lexical.joblib) stores the best model using lexical features.
  - [`best_model_semantic.joblib`](./models/best_model_semantic.joblib) stores the best model using semantic features.
  - [`best_model_stylistic.joblib`](./models/best_model_stylistic.joblib) stores the best model using stylistic features.
  - [`best_model_syntactic.joblib`](./models/best_model_syntactic.joblib) stores the best model using syntactic features.
- [`results/`](./results/) contains results and analysis outputs.
  - [`model_results.csv`](./results/model_results.csv) contains the final results from models.
  - [`test_features.csv`](./results/test_features.csv) contains extracted features for the test set.
  - [`train_features.csv`](./results/train_features.csv) contains extracted features for the training set.
- [`scripts/`](./scripts/) contains the main scripts and feature extraction modules.
  - [`FE_subscripts/`](./scripts/FE_subscripts/) contains the feature extraction modules.
    - [`lexical_features.py`](./scripts/FE_subscripts/lexical_features.py) implements lexical feature extraction.
    - [`semantic_features.py`](./scripts/FE_subscripts/semantic_features.py) implements semantic feature extraction.
    - [`stylistic_features.py`](./scripts/FE_subscripts/stylistic_features.py) implements stylistic feature extraction.
    - [`syntactic_features.py`](./scripts/FE_subscripts/syntactic_features.py) implements syntactic feature extraction.
    - [`feature_utils.py`](./scripts/FE_subscripts/feature_utils.py) provides utility functions for feature extraction.
    - [`submodels.py`](./scripts/FE_subscripts/submodels.py) implements submodels for experimentation.
  - [`data_loader.py`](./scripts/data_loader.py) loads the datasets.
  - [`experiments.py`](./scripts/experiments.py) runs the experimental pipeline for training models.
  - [`feature_analysis.py`](./scripts/feature_analysis.py) analyzes extracted features.
  - [`feature_extraction.py`](./scripts/feature_extraction.py) contains the main feature extraction logic.
- [`Experiments.ipynb`](./Experiments.ipynb) contains the pipeline for feature extraction and training.
- [`Results_and_Analysis.ipynb`](./Results_and_Analysis.ipynb) contains the pipeline for analyzing results.
- [`README.md`](./README.md) contains the project documentation.
- [`requirements.txt`](./requirements.txt) lists the required Python libraries.

## Features

The assignment constrained us to at least three dimensions (Lexical, Syntactical, Combined), but we decided to expand this to the following five: 
- **Lexical Features**
- **Syntactic Features**
- **Semantic Features** 
- **Stylistic Features** 
- **Combined Features** 

For all the details on what features we implemented and how, please see Section 2 in the [Results and Analysis notebook](./Results_and_Analysis.ipynb)

## Datasets

The datasets are located in the [`data/`](./data/) directory and consist of:
- **Train Data**: Used for training the models.
- **Test-Gold Data**: Used for evaluation against gold-standard similarity scores.

## Results

The results are saved in the [`results/`](./results/) directory, including:
- Extracted features ([`train_features.csv`](./results/train_features.csv), [`test_features.csv`](./results/test_features.csv)).
- Model performance ([`model_results.csv`](./results/model_results.csv)).

## Evaluation

The system is evaluated using **Pearson correlation**, as per SemEval 2012 guidelines. A comparison with official results from the SemEval task is also included in the analysis.
