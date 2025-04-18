# 🍷 Red Wine Quality Prediction

Predict the quality of red wine using machine learning based on various physicochemical tests. This project demonstrates data cleaning, feature engineering, model training, evaluation, and deployment of a predictive system.

## 📌 Project Overview

This project uses the **Red Wine Quality Dataset** to classify wine as high or low quality based on features like acidity, sugar level, alcohol, and more. The goal is to explore the relationships among features and use ML techniques to predict quality accurately.

---

## ⚙️ Tech Stack

- **Programming Language:** Python 3.9+
- **Data Manipulation:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Modeling & Evaluation:** scikit-learn (Random Forest, GridSearchCV)
- **Model Persistence:** pickle
- **Version Control & Deployment:** Git, GitHub

---

## 📁 Dataset

- **File:** `winequality-red.csv`
- **Rows:** 1599 samples
- **Target Variable:** `quality` (integer score between 0 and 10)

---

## Data Preparation

1. **Missing Value Check**
   - Ensured no null values were present in the dataset.

2. **Univariate Analysis**
   - Calculated summary stats for each numerical feature.
   - Visualized distributions.

3. **Outlier Treatment**
   - Used IQR method to detect and cap extreme values.

4. **Skewness Correction**
   - Applied `log1p` transformation on highly skewed features such as:
     - `residual sugar`
     - `chlorides`
     - `sulphates`

5. **Feature & Target Setup**
   - Feature selection via model-based filtering (`SelectFromModel`).
   - Target (`quality`) converted to binary classification:
     - **Good Quality**: `quality >= 7`
     - **Not Good**: `quality < 7`

---

