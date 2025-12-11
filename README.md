# DeepFake Detection Project

A comprehensive machine learning project to detect deepfake videos with high accuracy and explainability.

## 📋 Project Overview

**Objective**: Develop a model to detect deepfake videos with highest accuracy (possible by us) which has explainability. Will try to create few models to generate comparisons for comparison and then picking one final one as our "final model".

**Business Value**: Flagging misinformation / Protecting digital identity

**Dataset**: [Hemgg/deep-fake-detection-dfd-entire-original-dataset](https://huggingface.co/datasets/Hemgg/deep-fake-detection-dfd-entire-original-dataset)

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install torch torchvision torchaudio
pip install optuna timm datasets
pip install opencv-python scikit-image
pip install streamlit plotly pandas numpy matplotlib seaborn
pip install tqdm pillow
```

### 2. Run Jupyter Notebook

```bash
jupyter notebook DeepFakeDetection.ipynb
```

### 3. Run Streamlit Dashboard

```bash
streamlit run DeepFakeDetection_Dashboard.py
```

## 📁 Project Structure

```
Final Project/
├── DeepFakeDetection.ipynb          # Main analysis notebook
├── DeepFakeDetection_Dashboard.py   # Interactive Streamlit dashboard
├── create_notebook.py               # Notebook creation script
├── README.md                         # This file
└── requirements.txt                  # Python dependencies (to be created)
```

## 🔬 Approach

### 1. Data Loading & Exploration
- Load dataset from HuggingFace (200 samples subset)
- Analyze dataset structure and class distribution
- Visualize sample images/videos

### 2. Feature Engineering
- **Spatial Features**: Mean, std, variance, histogram features
- **Frequency Features**: FFT analysis
- **Texture Features**: LBP (Local Binary Pattern), GLCM
- **Color Features**: RGB, HSV, LAB statistics

### 3. Model Architectures
- **Simple CNN**: Lightweight baseline model
- **ResNet18**: Transfer learning with pretrained ImageNet weights
- **EfficientNet**: Efficient architecture for deployment

### 4. Training & Evaluation
- Train models with data augmentation
- Comprehensive evaluation metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
- Training history visualization
- Confusion matrices and ROC curves

### 5. Hyperparameter Tuning
- Automated optimization using Optuna
- Search space: Learning rate, batch size, weight decay, dropout
- Tree-structured Parzen Estimator (TPE) optimization

### 6. Model Comparison
- Compare all models across multiple metrics
- Select best performing model
- Analyze trade-offs

## 📊 Key Results

- **Best Model**: Optimized ResNet18
- **Accuracy**: ~100% on test set
- **All Metrics**: Perfect scores (1.0) for Precision, Recall, F1-Score, ROC-AUC

## 🎯 Features

- **Multiple Architectures**: Simple CNN, ResNet18, EfficientNet
- **Feature Engineering**: Handcrafted features (spatial, frequency, texture, color)
- **Hyperparameter Optimization**: Automated tuning with Optuna
- **Comprehensive Evaluation**: Multiple metrics and visualizations
- **Interactive Dashboard**: Streamlit app explaining all models and graphs
- **Model Explainability**: Feature importance, SHAP values, Grad-CAM

## 📚 Documentation

The Streamlit dashboard (`DeepFakeDetection_Dashboard.py`) provides comprehensive explanations of:
- Dataset overview and EDA
- Feature engineering methods
- Model architectures
- Training and evaluation results
- Hyperparameter tuning process
- Model comparison and selection
- Model explainability

## 👤 Author

Poonam and Mohini - ML PostGrad - Deep Learning Final Project

## 📝 License

This project is part of academic research.

