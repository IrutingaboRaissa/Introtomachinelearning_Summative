# Obesity Classification: Machine Learning vs Deep Learning

A comprehensive comparative analysis of classical machine learning and deep learning approaches for predicting obesity levels based on eating habits and physical conditions.

## Project Overview

This project implements and compares multiple machine learning algorithms to classify obesity levels using lifestyle, dietary, and demographic factors. The analysis demonstrates the effectiveness of various approaches from traditional ML to deep neural networks.

## Objectives

- Compare classical ML algorithms (Random Forest, SVM, Gradient Boosting, etc.) with deep learning models
- Evaluate model performance using comprehensive metrics (accuracy, precision, recall, F1-score)
- Perform extensive hyperparameter optimization using GridSearchCV and Optuna
- Analyze feature importance and model interpretability
- Provide insights for obesity prediction in clinical settings

## Dataset

**Source:** "Estimation of Obesity Levels Based On Eating Habits and Physical Condition"
- **Size:** 2,111 records
- **Features:** 17 attributes including age, gender, height, weight, dietary habits, physical activity, and lifestyle factors
- **Target:** Obesity classification (Underweight to Obesity Type III)
- **Split:** 70% training, 15% validation, 15% testing (stratified)

## Technologies Used

- **Python 3.x**
- **Libraries:** 
  - Scikit-learn (ML algorithms)
  - TensorFlow/Keras (Deep Learning)
  - Pandas, NumPy (Data manipulation)
  - Matplotlib, Seaborn (Visualization)
  - Optuna (Hyperparameter optimization)

## Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn kagglehub optuna
```

### Running the Notebook

1. Clone this repository
2. Open `summative.ipynb` in Jupyter Notebook or VS Code
3. Run cells sequentially from top to bottom
4. The notebook will automatically download the dataset from Kaggle

## Models Implemented

### Classical Machine Learning
- Random Forest Classifier
- Support Vector Machine (SVM)
- Gradient Boosting Classifier
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Naive Bayes
- Decision Tree
- Ensemble Methods (Voting, Bagging, AdaBoost)

### Deep Learning
- Multi-layer Neural Networks with various architectures
- Regularization techniques (Dropout, L1/L2)
- Advanced optimization (Adam, RMSprop, SGD)

## Key Features

- **Data Preprocessing:** Missing value handling, feature scaling, encoding
- **Feature Engineering:** Selection and importance analysis
- **Model Optimization:** GridSearchCV, RandomizedSearchCV, Optuna
- **Evaluation:** Cross-validation, confusion matrices, ROC curves, precision-recall analysis
- **Visualization:** Comprehensive plots and performance comparisons

## Results

The notebook provides detailed performance comparisons across all models with:
- Accuracy scores
- Precision, Recall, and F1-scores
- ROC-AUC curves
- Confusion matrices
- Training/validation curves
- Feature importance rankings

## Key Findings

Results demonstrate the comparative effectiveness of different approaches for obesity classification, with insights on:
- Model performance trade-offs
- Feature importance in obesity prediction
- Computational efficiency considerations
- Practical applicability in healthcare settings

## References

1. World Health Organization. (2021). Obesity and overweight fact sheet
2. Palechor, F. M., & de la Hoz Manotas, A. (2019). Dataset for estimation of obesity levels

## Author

**Irutingabo Raissa**

## License

This project is for educational purposes as part of an Introduction to Machine Learning summative assessment.
