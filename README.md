# Midterm Project for 'Machine-Learning-Zoomcamp'

## Introduction
Predicting accurate delivery times is critical for enhancing customer experience in food delivery services. This project focuses on building a machine learning model to estimate the delivery duration for DoorDash orders. The target metric is the total time, in minutes, between when a customer places an order and when the order is delivered.

The dataset contains a subset of DoorDash deliveries from early 2015, including features about time, stores, orders, and market conditions. For further details about the dataset, see [DoorDash ETA Prediction](https://www.kaggle.com/datasets/dharun4772/doordash-eta-prediction)

## Setup & Project Details

### Local Setup

1. **Environment Setup**:
   Create a Python virtual environment (Python >= 3.10 recommended) and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. **Preliminary Analysis**:
   Preliminary exploration work is in `notebook.ipynb`, which includes:

   - **Data Cleaning & EDA**:
      - Check the number of missing values in each column and clean the dataset.
      - Visualize feature distributions using `seaborn`.
      - Analyze linear correlations between numeric features and the target variable.

   - **Feature Engineering & Modeling & Hyperparameter Tuning**:
      - Categorical variables are encoded with `TargetEncoder` (`category_encoders` library).
      - Train and evaluate Random Forest and XGBoost models.
      - Use `Pipeline` and `GridSearchCV` from `scikit-learn` for hyperparameter tuning.
      - Results show that **XGBoost** outperformed Random Forest with lower prediction error (measured by mean absolute error).

3. **Model Training**:
   Train the final XGBoost model using the optimal hyperparameters obtained from grid search and save the complete pipeline for inference:

   ```bash
   python train.py
   ```

4. **Launch Web Service**:
   Run the prediction service using Streamlit:

   ```bash
   streamlit run predict.py
   ```

   Access the service at [http://localhost:8501/](http://localhost:8501/).

---

### Run with Docker

1. **Build the Docker Image**:
   ```bash
   docker build -t ml-project .
   ```

2. **Run the Docker Container**:
   ```bash
   docker run -p 8501:8501 ml-project
   ```

   The web service will be available at [http://localhost:8501/](http://localhost:8501/).

---