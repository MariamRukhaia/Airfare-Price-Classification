# ✈️ Airfare Price Classification

A machine learning pipeline for predicting whether an airline ticket's base fare is **above or below the national average**, using historical flight data and multiple classification approaches.

This project explores the full machine learning workflow — from large-scale data preprocessing and feature engineering to exploratory analysis, dimensionality reduction, model training, hyperparameter tuning, and evaluation.

---

## 🎯 Project Overview

Airfare pricing depends on a combination of factors including travel distance, flight duration, airline, route, booking time, and ticket characteristics.

The goal of this project was to formulate airfare prediction as a **binary classification problem**:

> **Can we predict whether an airline ticket's base fare is above or below the national average fare of $397.64?**

To investigate this problem, we compared three machine learning approaches:

- **Logistic Regression**
- **Support Vector Machines (SVM)**
- **Neural Networks**

We also explored the structure of the flight data using **K-Means clustering** and **Principal Component Analysis (PCA)**.

---

## 📊 Dataset

The project uses a Kaggle airfare dataset containing approximately:

- **6 million flight records**
- **27 original features**
- Flight searches and pricing information from Expedia

Features include information such as:

- Starting and destination airports
- Airline
- Search date and flight date
- Travel duration
- Total travel distance
- Basic economy status
- Refundability
- Nonstop status
- Flight segment information
- Base fare

Because of the size of the original dataset and repeated observations created by changing ticket prices, the data was sampled before model training.

> **Note:** The original dataset is not included in this repository due to its size. The project uses the [Flight Prices dataset from Kaggle](https://www.kaggle.com/datasets/dilwong/flightprices).

---

## ⚙️ Data Preprocessing & Feature Engineering

The raw flight data required substantial preprocessing before it could be used for machine learning.

### Data Preparation

- Sampled records from the multi-million-row dataset to reduce computational cost
- Shuffled observations before model training
- Removed records with missing values in critical features
- Removed identifiers and features that did not contribute useful predictive information

### Feature Engineering

Several new features were extracted or transformed:

- Converted boolean features such as `isBasicEconomy`, `isRefundable`, and `isNonStop` into binary values
- Converted flight and search dates into datetime representations
- Created `daysUntilFlight` from the difference between search and departure dates
- Extracted `searchDayOfWeek` and `flightDayOfWeek`
- One-hot encoded starting airports and destination airports
- Extracted and encoded airline information
- Aggregated multi-segment flight durations
- Aggregated multi-segment travel distances
- Standardized numerical features using `StandardScaler`

The target variable, `isAboveAvg`, was constructed from the ticket's base fare:

```text
0 → Base fare at or below $397.64
1 → Base fare above $397.64
```

The processed data was divided into:

```text
70% Training
15% Validation
15% Testing
```

---

## 🔍 Exploratory & Unsupervised Analysis

Before training the classification models, we explored relationships within the dataset using visualization and unsupervised learning.

### Exploratory Analysis

Feature distributions, scatter plots, and a correlation matrix were used to examine relationships between flight characteristics and airfare.

One of the clearest patterns was that **longer travel distances and durations were associated with a greater likelihood of an above-average ticket price**.

### K-Means + PCA

**K-Means clustering** was applied to the standardized training data to investigate whether natural groupings existed within the flight records.

Because the feature space was high-dimensional, **Principal Component Analysis (PCA)** was used to project the clusters into two dimensions for visualization.

This analysis showed some visible structure within the processed flight data and helped motivate the supervised classification experiments.

---

## 🤖 Machine Learning Models

### 1. Logistic Regression

Logistic Regression served as a baseline linear classifier.

Experiments included:

- Untransformed features
- Degree-2 polynomial transformation
- Degree-3 polynomial transformation
- Degree-3 transformation with PCA
- L2 regularization
- Multiple values of the regularization parameter `C`

The best validation performance was achieved using a **degree-2 feature transformation**.

**Best validation accuracy: 84.68%**

Increasing model complexity initially improved performance, but larger values of `C` and more complex transformations eventually produced overfitting.

---

### 2. Support Vector Machine

Support Vector Machines were evaluated using multiple kernel configurations:

- Linear / untransformed
- RBF kernel
- Polynomial degree 2
- Polynomial degree 3

Multiple values of `C` were evaluated to study the effect of regularization on model performance.

The strongest validation result was achieved using an:

**RBF Kernel with C = 10**

**Best validation accuracy: 85.48%**

Performance decreased at larger values of `C`, demonstrating the tradeoff between model complexity and generalization.

---

### 3. Neural Network

A Multi-Layer Perceptron was used to explore a nonlinear neural-network approach.

Experiments compared:

- Different hidden-layer architectures
- ReLU and Tanh activation functions
- Different regularization strengths (`alpha`)

The strongest configuration used:

```text
Hidden Layers: (100, 50)
Activation: ReLU
```

**Best validation accuracy: 85.34%**

Increasing regularization initially improved generalization, while excessively large `alpha` values caused the network to underfit.

---

## 📈 Results

After model selection and validation, the selected models were evaluated on the held-out test set.

| Model | Test Accuracy |
|---|---:|
| **Support Vector Machine** | **78.96%** |
| Logistic Regression | 77.21% |
| Neural Network | 73.43% |

The SVM achieved the highest test accuracy among the evaluated models.

Validation experiments also demonstrated that the highest validation accuracy did not necessarily translate directly to test performance, highlighting the importance of evaluating model generalization on unseen data.

---

## 🧠 Key Findings

- Travel **distance and duration** showed some of the strongest relationships with airfare.
- Feature engineering substantially expanded the information available to the models from the original flight records.
- Nonlinear transformations improved Logistic Regression beyond the untransformed baseline.
- The **RBF SVM** achieved the strongest validation performance at **85.48%**.
- Increasing model complexity could improve training performance while simultaneously reducing validation performance due to overfitting.
- Neural-network performance was sensitive to architecture and regularization strength.
- Final test performance was lower than peak validation performance, reinforcing the importance of maintaining a completely unseen test set.

---

## 🛠️ Tech Stack

`Python` `Pandas` `NumPy` `scikit-learn` `Matplotlib` `Seaborn` `Jupyter Notebook`

### Machine Learning

`Logistic Regression` `SVM` `MLP Neural Networks` `K-Means` `PCA` `Polynomial Features`

### Evaluation

`Accuracy` `Precision` `Recall` `F1 Score`

## 🚀 Running the Project

### 1. Clone the repository

```bash
git clone <repository-url>
cd Airfare-Price-Classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
notebooks/airfare-price-classification.ipynb
```

The original airfare dataset must be downloaded separately from the [Kaggle Flight Prices dataset](https://www.kaggle.com/datasets/dilwong/flightprices) and placed in the appropriate local data directory before running the full preprocessing pipeline.


## 👥 Authors

**Mariam Rukhaia**  
**Oleg Vengrovych**
