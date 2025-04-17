# ❤️ Heart Disease Prediction Model

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Implementation](#model-implementation)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## 🌟 Project Overview

This project implements a machine learning model to predict the likelihood of heart disease based on various medical attributes. The model uses Logistic Regression, a classification algorithm ideal for binary outcome prediction, to determine whether a patient is at risk of heart disease.

The primary goal is to provide an accurate tool that can assist healthcare professionals in early detection and risk assessment of heart conditions, potentially saving lives through timely intervention.

## 📊 Dataset

The dataset used for training and evaluation is sourced from a comprehensive collection of patient records with heart-related symptoms and diagnoses.

[Heart Disease Dataset Link](https://drive.google.com/file/d/1aSOlyPFPL-Ocy0cHPFof1fg8izkuj_LU/view?usp=drive_link)

### Dataset Features

The dataset includes the following attributes:
- Demographic information (age, sex)
- Clinical measurements (blood pressure, cholesterol levels)
- Cardiac test results (resting ECG, maximum heart rate)
- Symptom information (chest pain type, angina)
- Other relevant medical indicators

## 🧠 Model Implementation

### Logistic Regression Model

This project implements a Logistic Regression model - a statistical method that analyzes the relationship between multiple independent variables and a categorical dependent variable. For heart disease prediction, the model:

- Analyzes patient attributes to estimate heart disease probability
- Uses a sigmoid function to transform predictions into probability values (0-1)
- Classifies patients based on a threshold probability value
- Optimizes the decision boundary to maximize prediction accuracy

### Data Preprocessing Steps

- Handling missing values
- Feature scaling and normalization
- Encoding categorical variables
- Feature selection and engineering
- Train-test splitting for model validation

## 📥 Installation

```bash
# Clone the repository
git clone https://github.com/vinayakjoshi04/Heart-Disease.git

# Navigate to the project directory
cd Heart-Disease

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt

# Open the Jupyter notebook
jupyter notebook heart_diesease.ipynb
```

## 🚀 Usage

1. Open the Jupyter notebook `heart_diesease.ipynb`
2. Follow the step-by-step implementation and analysis
3. To use the trained model for predictions:

```python
# Load the trained model
import pickle
with open('heart_disease_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Prepare input data (example)
# Features must match the training data format
input_data = [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]

# Make prediction
prediction = model.predict(input_data)
probability = model.predict_proba(input_data)

print(f"Heart Disease Prediction: {'Positive' if prediction[0] == 1 else 'Negative'}")
print(f"Probability: {probability[0][1]:.2f}")
```

## 📈 Results

The Logistic Regression model achieved the following performance metrics:

- **Accuracy**: 49%

### Model Validation
The model was validated using k-fold cross-validation to ensure reliable performance across different data subsets.

### Feature Importance
Analysis of feature coefficients revealed the most significant predictors of heart disease:
- [List top features]

## 📁 Project Structure

```
Heart-Disease/
│
├── heart_diesease.ipynb     # Main Jupyter notebook with analysis and model implementation
├── heart_disease_model.pkl  # Serialized trained model (if available)
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
│
└── data/                    # Data directory (if used)
    └── heart_disease.csv    # Dataset (if included)
```

## 🔮 Future Improvements

- Implement ensemble methods for improved accuracy
- Develop a web application interface for easy access
- Add visualization tools for better interpretability
- Include additional biomarkers to enhance prediction accuracy
- Explore deep learning approaches for complex pattern recognition

## 👥 Contributing

Contributions to improve the model or extend its functionality are welcome:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

Developer: [Vinayak Joshi](https://github.com/vinayakjoshi04)

For questions or collaboration opportunities, please reach out through GitHub.

## 🙏 Acknowledgments

- Thanks to all the healthcare professionals who provided domain expertise
- Special appreciation to the open-source community for providing valuable tools and libraries
