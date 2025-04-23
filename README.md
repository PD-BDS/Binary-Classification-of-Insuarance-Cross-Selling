# 🧾 Insurance Cross Selling Prediction

This project tackles the problem of predicting customer responses to a cross-selling offer in the insurance sector. Using an imbalanced dataset, the notebook explores data preprocessing, feature engineering, and deep learning modeling to classify whether a customer will be interested in additional insurance products.

📊 **Competition Dataset**: [Kaggle Playground Series - Season 4, Episode 7](https://www.kaggle.com/competitions/playground-series-s4e7)

---

## 🧩 Project Highlights

- 💼 Business Problem: Predict if a customer will be interested in vehicle insurance based on their demographic and behavioral features.
- 📊 Dataset: 38,000+ customer records with demographic and policy details.
- 🔄 Handled severe class imbalance using ADASYN.
- 🧠 Modeled with a custom-built Multilayer Perceptron (MLP) using TensorFlow/Keras.
- 🔍 Extensive hyperparameter tuning (layers, activation functions, etc.)

---

## 📁 Project Structure

```
Insurance_Cross_Selling/
│
├── Insurance_cross_selling.ipynb    # Full workflow in notebook
└── README.md                        # You're here!
```

---

## ⚙️ Techniques Used

### 📌 Data Preprocessing

- Removed duplicates and nulls
- Feature engineering on `Age` (binned into groups)
- One-hot encoding of categorical variables
- Normalization using `MinMaxScaler`

### 📊 Imbalanced Classification

- Used **ADASYN (Adaptive Synthetic Sampling)** to oversample the minority class.
- Pre-ADASYN accuracy: ~85% (but poor minority class detection)
- Post-ADASYN accuracy: ~70% (better minority recall)

### 🧠 MLP Model

- Built using Keras Sequential API
- Layers experimented with:
  - Dense(64) → ReLU
  - Dense(128) → ReLU
  - Dense(64) → ReLU
  - Dropout(0.2)
  - Dense(1) → Sigmoid
- Loss: `binary_crossentropy`
- Optimizer: `adam`

### 📈 Evaluation Metrics

- Accuracy
- ROC-AUC Score
- Precision, Recall, F1-score
- Confusion Matrix

---

## 🧪 Results & Observations

- The dataset was highly skewed (~90% negative class).
- Without balancing, the model was biased toward the majority class.
- ADASYN successfully improved recall and F1 for the minority class.
- Future improvements could include:
  - Using different sampling techniques (SMOTE, Borderline-SMOTE)
  - Trying other architectures (CNN, XGBoost)
  - Incorporating cost-sensitive loss functions

---

## 🚀 Running the Notebook

To reproduce the results:

1. Install necessary packages:
   ```bash
   pip install pandas scikit-learn imbalanced-learn matplotlib seaborn tensorflow
   ```

2. Open the notebook:
   - Run in [Google Colab](https://colab.research.google.com)
   - Or locally in Jupyter Notebook

3. Download the dataset from Kaggle:
   - Upload `kaggle.json`
   - Notebook handles automatic download and extraction


---

## 🙌 Acknowledgements

- [Kaggle](https://www.kaggle.com/)
- [Imbalanced-learn](https://imbalanced-learn.org/stable/)
- [TensorFlow](https://www.tensorflow.org/)
- [Google Colab](https://colab.research.google.com/)
