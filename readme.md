# 🏠 Real Estate Price Predictor – Project Description

## 📌 Overview
The **Real Estate Price Predictor** is a machine learning application that uses **linear and polynomial regression** to predict house prices based on various features like square footage, number of bedrooms, age of the house, and location score. The project demonstrates key ML concepts such as **feature vectorization, scaling, cost optimization, learning rate tuning**, and **model training automation** using **GitHub Actions CI/CD pipelines**.

---

## 🎯 Objectives
- Implement a **linear regression model** from scratch using NumPy.
- Apply **feature engineering**, **vectorization**, and **scaling** to prepare data.
- Explore **polynomial regression** for better fit in non-linear cases.
- Track training performance using a **cost function (MSE)**.
- Tune the **learning rate** and visualize training convergence.
- Create a reproducible workflow with **GitHub Actions** to automate training, testing, and model versioning.

---

## 📊 Dataset
Use a synthetic or real-world dataset with the following columns:
- `size_sqft` – size of the house in square feet
- `num_bedrooms` – number of bedrooms
- `age` – age of the property
- `location_score` – encoded location quality (0–10)
- `price` – target price in USD

You can use:
- A cleaned version of the [Boston Housing Dataset](https://www.kaggle.com/c/boston-housing)
- Or generate synthetic data using NumPy and Pandas.

---

## 📐 Core ML Concepts Implemented

### 1. **Vectorization**
- Use NumPy arrays instead of loops to compute predictions and gradients.
- Demonstrate speedup from vectorized vs. loop-based implementations.

### 2. **Feature Scaling**
- Implement **standardization (z-score normalization)**:

### 3. **Cost Function**
- Use **Mean Squared Error (MSE)**

### 4. **Gradient Descent**
- Train the model using gradient descent.
- Show effect of **learning rate** on convergence.
- Plot the **cost vs. iterations** graph.

### 5. **Polynomial Regression**
- Extend features to quadratic terms: `x1^2`, `x1*x2`, `x2^2`, etc.
- Compare linear vs. polynomial fit (using 2D/3D plots).

### 6. **Model Evaluation**
- Evaluate using:
  - Training/test split
  - R² score
  - Residual plots

---

## ⚙️ DevOps Integration

### GitHub Actions CI/CD
Automate:
- Code linting (e.g., with `flake8`)
- Unit tests for:
  - Cost function
  - Gradient computation
  - Prediction accuracy
- Model training
- Save trained model artifact 
---

## 📚 Deliverables
- ✅ Clean codebase with docstrings and comments
- ✅ GitHub repo with commit history and GitHub Actions
- ✅ Report or README with:
  - Model explanation
  - Graphs: cost curve, residuals, R² score
  - Comparison of linear vs. polynomial regression
- ✅ Working regression model with training script
