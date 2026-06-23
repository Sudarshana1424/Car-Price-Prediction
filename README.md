# Car Price Prediction using Machine Learning

## Overview

This project predicts the selling price of used cars using Machine Learning. It demonstrates an end-to-end ML pipeline including data preprocessing, feature engineering, exploratory data analysis (EDA), model training, and performance evaluation.

The project compares the performance of **Linear Regression** and **Random Forest Regressor** to identify the best-performing model.

---

## Features

- Data cleaning and preprocessing
- Missing value handling
- Feature engineering
- Exploratory Data Analysis (EDA)
- One-Hot Encoding of categorical variables
- Feature Scaling using StandardScaler
- Linear Regression model
- Random Forest Regressor model
- Model evaluation using R² Score

---

## Dataset

The dataset contains information about used cars, including:

- Car Name
- Manufacturer
- Year
- Kilometers Driven
- Fuel Type
- Transmission
- Owner Type
- Mileage
- Engine Capacity
- Power
- Seats
- Selling Price (Target Variable)

Place the dataset inside:

```
data/
    dataset.csv
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Project Structure

```
Car-Price-Prediction/
│
├── data/
│   └── dataset.csv
│
├── car_price_prediction.py
├── requirements.txt
└── README.md
```

---

## Data Preprocessing

The project performs several preprocessing steps:

- Removes unnecessary columns
- Extracts manufacturer from car name
- Converts manufacturing year into vehicle age
- Cleans numerical values from:
  - Mileage
  - Engine
  - Power
- Handles missing values using mean imputation
- Drops unavailable columns
- Performs One-Hot Encoding
- Aligns train and test features
- Standardizes numerical features

---

## Exploratory Data Analysis

The project includes visualization of:

- Distribution of manufacturers
- Vehicle counts by manufacturer

Additional visualizations can easily be added for deeper analysis.

---

## Machine Learning Models

### 1. Linear Regression

A baseline regression model used for predicting car prices.

### 2. Random Forest Regressor

An ensemble learning model that generally provides better prediction accuracy by combining multiple decision trees.

---

## Model Evaluation

Performance is evaluated using the **R² Score**.

```python
from sklearn.metrics import r2_score

r2_score(y_test, predictions)
```

A higher R² score indicates better prediction performance.

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Car-Price-Prediction.git
```

Move into the project directory

```bash
cd Car-Price-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the project

```bash
python car_price_prediction.py
```

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

Or install manually:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## Future Improvements

- Hyperparameter tuning
- Cross-validation
- XGBoost/CatBoost implementation
- Feature importance visualization
- Model deployment using Flask or FastAPI
- Interactive web application using Streamlit

---

## Results

The project compares the predictive performance of:

- Linear Regression
- Random Forest Regressor

The Random Forest model generally achieves better accuracy due to its ability to capture non-linear relationships within the data.

---

## License

This project is open-source and available under the MIT License.
