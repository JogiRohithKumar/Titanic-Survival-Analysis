# Machine Learning Data Preprocessing Pipeline
Data cleaning and preprocessing of the Titanic dataset


## Titanic Dataset Preprocessing Project

This project was built as part of my machine learning learning journey. This project focuses on **cleaning** and **preparing** the Titanic dataset for **machine learning** analysis. The main objective is to **handle missing data**, **convert categorical variables to numeric**, **normalize numerical features**, and **detect/remove outliers** to ensure the data is ready for modeling.

---

## 📂 Dataset Overview

The Titanic dataset contains information about the passengers aboard the Titanic ship, including:

- **PassengerId**  
- **Survived** (target variable)  
- **Pclass** (ticket class)  
- **Name**  
- **Sex**  
- **Age**  
- **SibSp** (siblings/spouses aboard)  
- **Parch** (parents/children aboard)  
- **Ticket**  
- **Fare**  
- **Cabin**  
- **Embarked** (port of embarkation)  

---

## ✔ Steps Performed

### 1. Dataset Import & Initial Exploration
- Loaded the dataset using **pandas**.  
- Checked dataset shape, data types, and summary statistics.  
- Identified missing values using `.isnull().sum()`.  
- Analyzed categorical variables using `.value_counts()`.  

### 2. Handling Missing Values
- Imputed missing **Age** values using the **median**.  
- Filled missing **Embarked** entries using the most frequent category (**mode**).  
- Dropped **Cabin** column due to excessive missing data.  

### 3. Encoding Categorical Variables
- Converted **Sex** column into numerical values using **label encoding** (`male`=1, `female`=0).  
- Applied **one-hot encoding** to the **Embarked** column to handle multiple categories.  

### 4. Feature Scaling
- Used **StandardScaler** from scikit-learn to **standardize** the **Age** and **Fare** features (mean = 0, std = 1).  
- Scaling improves model performance by **normalizing feature ranges**.  

### 5. Outlier Detection and Removal
- Visualized outliers in **Age** and **Fare** using **boxplots** from seaborn.  
- Applied the **Interquartile Range (IQR)** method to detect and remove outliers:  
  - Calculated **Q1** (25th percentile) and **Q3** (75th percentile).  
  - Defined bounds as **Q1 - 1.5*IQR** and **Q3 + 1.5*IQR**.  
  - Filtered dataset to keep only data within these bounds.  

---

## Key Outcomes

✅ Cleaned 891 passenger records

✅ Engineered passenger title features

✅ Compared 9 machine learning models

✅ Achieved 83.5% cross-validation accuracy

✅ Built an end-to-end ML workflow

---

## 🛠️ Tools & Libraries Used

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
