# Neural-Network-Firefly-Genetic-Hybrid

# Markdown Note for Data Processing and Model Training

## Overview

This document outlines the process of loading, cleaning, preprocessing data, and implementing machine learning models using Firefly and Genetic Algorithms for feature selection and neural network optimization, respectively.

## Importing Necessary Libraries

```python
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
warnings.filterwarnings('ignore')
```

## Loading Data

```python
df = pd.read_csv('New_Data_Soft_Solution.csv')
df
```

## Data Exploration

- Check for unique status values:
  ```python
  df.status.unique()
  ```
- Check for missing values:
  ```python
  df.isna().sum()
  ```

## Data Cleaning

- Drop rows with missing 'dpa_nam' values:
  ```python
  df.dropna(subset=['dpa_nam'], inplace=True)
  df.isnull().sum()
  ```

- Drop unnecessary columns and fill missing values:
  ```python
  df.drop(columns=['nature_transaction', 'sgd_date', 'sgd_num'], inplace=True)
  df['selected_colour'].fillna('Unknown', inplace=True)
  df.fillna(0, inplace=True)
  ```

## Defining Features and Target

```python
X = df.drop(columns=['query_flag'])  # Features
y = df['query_flag']  # Target
```

## Splitting Data

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Preprocessing

- Separate non-numeric and numeric columns:
  ```python
  non_numeric_columns = ['importer_tin', 'declarant_code', 'office_cod', 'cty_origin',
                         'selected_colour', 'hscode', 'mot_cod', 'typ', 'status',
                         'dpa_nam', 'rat_customs_duty', 'prc_ext']
  numeric_columns = [col for col in X.columns if col not in non_numeric_columns]
  ```

- Label encode non-numeric columns:
  ```python
  def fit_and_transform_le(df_train, df_test, columns):
      for col in columns:
          le = LabelEncoder()
          df_train[col] = le.fit_transform(df_train[col].astype(str))
          test_categories = set(df_test[col].astype(str).unique())
          train_categories = set(le.classes_)
          unseen_categories = test_categories - train_categories
          if unseen_categories:
              le.classes_ = np.append(le.classes_, list(unseen_categories))
          df_test[col] = le.transform(df_test[col].astype(str))
      return df_train, df_test

  X_train, X_test = fit_and_transform_le(X_train, X_test, non_numeric_columns)
  ```

- Scale numeric columns:
  ```python
  scaler = StandardScaler()
  X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
  X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])
  ```

## Firefly Algorithm for Feature Selection

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def firefly_algorithm(X, y, feature_names, n_fireflies=20, max_iter=50, alpha=0.5, beta0=1, gamma=1):
    # Firefly Algorithm implementation
    # (Initialization, Attractiveness, Movement, Evaluation)
    # Select features using the best firefly
    return best_firefly, selection_percentages

# Feature names
feature_names = X_train.columns.values

# Run Firefly Algorithm
best_features, selection_percentages = firefly_algorithm(X_train.values, y_train.values, feature_names)
```

## Genetic Algorithm for Neural Network Optimization

```python
def create_nn(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def genetic_algorithm(X, y, input_dim, population_size=20, generations=10, mutation_rate=0.1):
    # Genetic Algorithm implementation
    # (Initialization, Evaluation, Crossover, Mutation)
    return best_model

# Select features using best firefly
selected_features_idx = np.where(best_features == 1)[0]
X_train_selected = X_train.iloc[:, selected_features_idx].values
X_test_selected = X_test.iloc[:, selected_features_idx].values

# Neural Network Optimization using GA
best_nn = genetic_algorithm(X_train_selected, y_train, input_dim=X_train_selected.shape[1])
```

## Model Evaluation

```python
# Evaluate the best neural network model on the test set
test_loss, test_accuracy = best_nn.evaluate(X_test_selected, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")
print(f'Test Loss: {test_loss}')

# Generate classification report and confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

# Predict probabilities and convert to binary predictions
y_pred = best_nn.predict(X_test_selected)
y_pred_class = (y_pred > 0.5).astype(int)

# Classification report
print("Classification Report:\n", classification_report(y_test, y_pred_class))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_class)
print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
```

## Results

- **Test Accuracy:** 0.9263
- **Test Loss:** 0.1820
- **Classification Report:**

  ```
               precision    recall  f1-score   support
          0.0       0.93      0.99      0.96     10166
          1.0       0.89      0.44      0.59      1403
      accuracy                           0.93     11569
     macro avg       0.91      0.72      0.78     11569
  weighted avg       0.92      0.93      0.92     11569
  ```

- **Confusion Matrix:**

  ```
  [[10092    74]
   [  779   624]]
  ```

## Conclusion

This workflow demonstrates how to preprocess data, select features using the Firefly Algorithm, and optimize a neural network using a Genetic Algorithm, leading to a high-accuracy model.
