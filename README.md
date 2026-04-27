# 🫀 Heart Disease Prediction using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A comprehensive machine learning project for predicting heart disease using ensemble methods with hyperparameter tuning and cross-validation

## 📊 About

This project implements and compares three powerful ensemble machine learning algorithms to predict the presence of heart disease in patients. The analysis includes extensive exploratory data analysis (EDA), feature engineering, outlier handling, and rigorous model evaluation using stratified k-fold cross-validation.

**Key Highlights:**
- ✨ Achieved **84.2% test accuracy** with Random Forest
- 🎯 **92.3% ROC-AUC score** demonstrating excellent classification performance
- 🔍 Comprehensive feature selection using correlation analysis and Chi-Square tests
- ⚙️ Hyperparameter optimization using GridSearchCV with 5000+ parameter combinations
- 📈 Stratified k-fold cross-validation ensuring robust model evaluation

## 🎯 Project Objectives

1. Perform comprehensive exploratory data analysis on heart disease dataset
2. Handle outliers and encode categorical features appropriately
3. Implement feature selection to identify most predictive variables
4. Train and optimize three ensemble models: Random Forest, Gradient Boosting, and XGBoost
5. Compare model performance using multiple metrics
6. Identify the best model for heart disease prediction

## 📁 Dataset Description

The dataset contains **303 patient records** with **14 features** related to heart health:

| Feature | Description | Type |
|---------|-------------|------|
| **age** | Age of the patient (years) | Continuous |
| **sex** | Gender (0 = female, 1 = male) | Binary |
| **cp** | Chest pain type (0-3) | Categorical |
| **trestbps** | Resting blood pressure (mm Hg) | Continuous |
| **chol** | Serum cholesterol (mg/dl) | Continuous |
| **fbs** | Fasting blood sugar > 120 mg/dl | Binary |
| **restecg** | Resting ECG results (0-2) | Categorical |
| **thalach** | Maximum heart rate achieved | Continuous |
| **exang** | Exercise-induced angina | Binary |
| **oldpeak** | ST depression induced by exercise | Continuous |
| **slope** | Slope of peak exercise ST segment (0-2) | Categorical |
| **ca** | Number of major vessels (0-4) | Discrete |
| **thal** | Thalium stress test result (0-3) | Categorical |
| **target** | Heart disease presence (0 = no, 1 = yes) | Binary |

**Dataset Statistics:**
- Total samples: 303
- No missing values
- Balanced target distribution (54.5% positive cases)

## 🛠️ Technologies & Libraries

```python
# Core Libraries
- Python 3.8+
- NumPy
- Pandas

# Visualization
- Matplotlib
- Seaborn

# Machine Learning
- scikit-learn
- XGBoost

# Key Modules Used
- RandomForestClassifier
- GradientBoostingClassifier
- XGBClassifier
- GridSearchCV
- StratifiedKFold
- StandardScaler
- SelectKBest (Chi-Square)
```

## 🚀 Installation & Setup

### Prerequisites
```bash
Python 3.8 or higher
pip package manager
```

### Clone Repository
```bash
git clone https://github.com/muhammedshihab1001/heart-disease-prediction.git
cd heart-disease-prediction
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Requirements File (`requirements.txt`)
- numpy>=1.21.0
- pandas>=1.3.0
- matplotlib>=3.4.0
- seaborn>=0.11.0
- scikit-learn>=1.0.0
- xgboost>=1.5.0

---

## 📂 Project Structure
```
heart-disease-prediction/
│
├── dataset/
│   └── heart.csv                 # Heart disease dataset
│
├── notebooks/
│   └── heart_disease_prediction.ipynb  # Main analysis notebook
│
├── images/                       # Generated visualizations
│   ├── distributions.png
│   ├── boxplots.png
│   ├── correlation_heatmap.png
│   ├── pairplot.png
│   ├── roc_curves.png
│   └── model_comparisons.png
│
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
└── LICENSE                       # License file
```

## 🔬 Methodology

### 1. **Exploratory Data Analysis (EDA)**
- Statistical summary of all features
- Distribution analysis of key variables
- Outlier detection using boxplots
- Correlation analysis with heatmap
- Pairplot visualization by target class

### 2. **Data Preprocessing**
- **Outlier Handling:** IQR-based capping for `trestbps`, `chol`, and `oldpeak`
- **Feature Encoding:** One-hot encoding for categorical variables (`cp`, `thal`, `slope`)
- **Feature Selection:** 
  - Correlation analysis with target variable
  - Chi-Square statistical test
  - Combined 12 most important features

### 3. **Model Training & Optimization**

#### **Random Forest Classifier**
```python
Best Parameters:
- n_estimators: 300
- max_depth: 6
- max_features: 'sqrt'
- min_samples_split: 2
- min_samples_leaf: 4
- class_weight: 'balanced'
```

#### **Gradient Boosting Classifier**
```python
Best Parameters:
- n_estimators: 200
- learning_rate: 0.1
- max_depth: 3
- subsample: 1.0
- max_features: 'sqrt'
```

#### **XGBoost Classifier**
```python
Best Parameters:
- n_estimators: 100
- learning_rate: 0.05
- max_depth: 3
- colsample_bytree: 0.8
- gamma: 0.2
- reg_alpha: 0.5
- reg_lambda: 2
- scale_pos_weight: 2
```

### 4. **Model Evaluation**
- Stratified 3-fold cross-validation
- 80-20 train-test split
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC

## 📊 Results

### Model Performance Comparison

| Model | Test Accuracy | CV Accuracy | ROC-AUC | Recall |
|-------|--------------|-------------|---------|--------|
| **Random Forest** | **84.2%** | 80.7% | **0.923** | 81% |
| Gradient Boosting | 78.9% | **81.6%** | 0.892 | 81% |
| XGBoost | 78.9% | **82.0%** | 0.921 | **90%** |

### Key Findings

🏆 **Best Overall Model: Random Forest**
- Highest test accuracy (84.2%)
- Best ROC-AUC score (0.923)
- Excellent precision-recall balance
- Most stable performance

🎯 **XGBoost Strengths:**
- Best cross-validation accuracy (82.0%)
- Highest recall (90%) - catches more positive cases
- Excellent for minimizing false negatives

📈 **Feature Importance:**
Top predictive features identified:
1. `thal_2` & `thal_3` (Thalium stress test results)
2. `oldpeak` (ST depression)
3. `exang` (Exercise-induced angina)
4. `thalach` (Maximum heart rate)
5. `ca` (Number of major vessels)

## 💡 Usage

### Running the Notebook
```bash
jupyter notebook notebooks/heart_disease_prediction.ipynb
```

### Making Predictions (Example)
```python
# Load the best model
import pickle
model = pickle.load(open('models/random_forest_best.pkl', 'rb'))

# Example patient data
patient_data = {
    'age': 55, 'ca': 0, 'thal_3': 1, 'exang': 0,
    'cp_2': 1, 'thalach': 150, 'slope_2': 1,
    'chol': 240, 'slope_1': 0, 'sex': 1,
    'oldpeak': 1.2, 'thal_2': 0
}

# Predict
prediction = model.predict([list(patient_data.values())])
probability = model.predict_proba([list(patient_data.values())])

print(f"Prediction: {'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'}")
print(f"Confidence: {max(probability[0]) * 100:.2f}%")
```

## 📈 Visualizations

The project includes comprehensive visualizations:

1. [**Distribution Plots**](./images/distributions.png) - Histograms with KDE for key features
2. [**Boxplots**](./images/boxplots.png) - Outlier detection and spread analysis
3. [**Correlation Heatmap**](./images/correlation_heatmap.png) - Feature relationships
4. [**Pairplot**](./images/pairplot.png) - Multivariate relationships by target class
5. [**ROC Curves**](./images/roc_curves.png) - Model discrimination ability
6. [**Performance Comparisons**](./images/model_comparisons.png) - Bar charts for accuracy and recall

## 🔮 Future Improvements

- [ ] Implement SMOTE for handling class imbalance
- [ ] Add deep learning models (Neural Networks)
- [ ] Feature engineering with polynomial features
- [ ] Ensemble stacking of multiple models
- [ ] Deploy model as REST API using Flask/FastAPI
- [ ] Create interactive dashboard with Streamlit
- [ ] Implement SHAP for model interpretability
- [ ] Add more datasets for cross-validation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Muhammed Shihab P**

`Machine Learning Enthusiast`

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the heart disease dataset
- scikit-learn and XGBoost communities for excellent documentation
- Kaggle community for inspiration and best practices

## 📚 References

1. UCI Heart Disease Dataset: [Link](https://archive.ics.uci.edu/ml/datasets/heart+disease)
2. Random Forest: Breiman, L. (2001). "Random Forests"
3. XGBoost: Chen & Guestrin (2016). "XGBoost: A Scalable Tree Boosting System"
4. Gradient Boosting: Friedman, J. H. (2001). "Greedy function approximation"

---

⭐ If you found this project helpful, please consider giving it a star!