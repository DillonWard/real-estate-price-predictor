# 🏠 Real Estate Price Predictor

## 📌 Overview
A machine learning project to predict real estate prices using linear and polynomial regression. The project demonstrates core ML concepts, robust engineering practices, and DevOps automation with GitHub Actions.

---

## 🎯 Objectives
- Build and compare linear and polynomial regression models for price prediction.
- Apply feature engineering, vectorization, and scaling.
- Visualize training progress and model fit.
- Automate training, testing, and model versioning with CI/CD.

---

## 📊 Dataset
Uses the [Boston Housing Dataset](https://www.kaggle.com/c/boston-housing) or synthetic data with features like:
- `size_sqft`, `num_bedrooms`, `age`, `location_score`, and target `price`.

---

## 🧠 ML Concepts Implemented

### 1. Vectorization
- Efficient NumPy-based computations for predictions and gradients.

### 2. Feature Scaling
- Standardization (z-score normalization) for improved model convergence.

### 3. Cost Function
- Mean Squared Error (MSE) for training and evaluation.

### 4. Gradient Descent
- Custom implementation with learning rate tuning and cost vs. iteration plots.

### 5. Polynomial Regression
- Feature expansion to quadratic terms for non-linear relationships.
- Visual comparison of linear vs. polynomial fits.

### 6. Model Evaluation
- Train/test split, R² score, and residual plots for performance assessment.

---

## ⚙️ DevOps Integration

### GitHub Actions CI/CD
Automates:
- Code linting (`flake8`)
- Unit tests (core functions, model accuracy)
- Coverage reporting
- Model training and artifact saving

Pre-commit hooks ensure code quality before every commit.

---

## 📚 Deliverables
- ✅ Clean, well-documented codebase
- ✅ Automated CI/CD pipeline
- ✅ Unit tests with coverage
- ✅ Model training scripts and Jupyter notebook
- ✅ Results table and visualizations (cost curves, residuals, R² scores)
- ✅ README/report with explanations and comparisons

---

## 🚀 Usage

1. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```

2. **Train models**
   ```sh
   python jobs.py
   ```

3. **Run tests**
   ```sh
   coverage run --source=src -m unittest tests.py
   coverage report
   ```

4. **View results**
   - See Jupyter notebook for plots and results table.
   - Trained models and parameters are saved in the `data/` directory.

---

## 🏆 Results

- Linear and polynomial regression compared on real/synthetic data.
- Automated workflow ensures reproducibility and code quality.
- Ready for deployment or further extension.

---

## 📬 Contact

For questions or collaboration, open an issue or pull request on GitHub.