# Titanic Survival Prediction Project

This project aims to predict the survival of passengers on the Titanic using machine learning techniques. The analysis involves data understanding, preprocessing, feature engineering, model training, hyperparameter tuning, and evaluation.

# Project Overview

The dataset contains information about Titanic passengers, including demographic and travel-related features. The goal is to build a model that predicts whether a passenger survived or not.

# Features Used 

PassengerId (dropped before modeling)

Survived (target variable)

Pclass (Passenger class)

Name (used for extracting titles, then dropped)

Sex (encoded as 0 for male, 1 for female)

Age (missing values imputed with median)

SibSp (number of siblings/spouses aboard)

Parch (number of parents/children aboard)

Ticket (dropped)

Fare (missing values imputed with median)

Cabin (dropped due to many missing values)

Embarked (one-hot encoded)

# Additional engineered features:

Title: Extracted from passenger names and mapped to numerical categories

FamilySize: Sum of SibSp and Parch plus one (self)

IsAlone: Binary feature indicating if the passenger was alone

Numerical features were normalized using StandardScaler.

# Data Preprocessing

Missing values in Age, Fare, and Embarked were imputed.

Cabin column was dropped due to excessive missing data.

Categorical variables (Sex, Embarked, Title) were encoded into numerical formats.

Feature scaling applied to numerical columns.

# Models Implemented 

Logistic Regression

Random Forest Classifier

Support Vector Machine (SVM)

Decision Tree Classifier

Gradient Boosting Classifier

Each model was trained and evaluated on accuracy, precision, recall, and F1 score.

# Model Selection and Hyperparameter Tuning

Random Forest was selected as the best-performing model and further optimized using GridSearchCV for hyperparameters such as number of estimators, max depth, and minimum samples per split/leaf.

# Results

All models achieved very high accuracy (close to or at 100%) on the test set, with Random Forest performing best after tuning.

Cross-validation confirmed the robustness of the Random Forest model with a mean accuracy of 1.0.

# Feature Importance

Feature importance was analyzed for the Random Forest model to understand which features contributed most to survival prediction.

# Model Persistence

The best Random Forest model was saved as titanic_survival_model.pkl using joblib for later use.

# How to Run
Install required packages: pandas, numpy, matplotlib, seaborn, scikit-learn, joblib

Load the dataset tested.csv into the working directory.

Run the notebook or script to preprocess data, train models, and evaluate results.

Use the saved model file for predictions on new data.

# Dependencies

Python 3.x

pandas

numpy

matplotlib

seaborn

scikit-learn

joblib
