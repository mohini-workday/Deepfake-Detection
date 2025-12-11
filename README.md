# DeepFake Detection Project

A comprehensive machine learning project to detect deepfake videos with high accuracy and explainability.

## 📋 Project Overview

**Objective**: Develop a model to detect deepfake videos with highest accuracy (possible by us) which has explainability. Will try to create few models to generate comparisons for comparison and then picking one final one as our "final model".

**Business Value**: Flagging misinformation / Protecting digital identity

**Dataset**: 
- **Source**: [Hemgg/deep-fake-detection-dfd-entire-original-dataset](https://huggingface.co/datasets/Hemgg/deep-fake-detection-dfd-entire-original-dataset)
- **Google Drive**: https://drive.google.com/open?id=1iLx76wsbi9itnkxSqz9BVBl4ZvnbIazj
- **Total Videos**: 6,075 training videos (4,860 train + 1,215 validation) + 15 test videos
- **Celeb-Real**: 493 real videos (Label: 0)
- **Celeb-Synthesis (Fake)**: 5,582 fake videos (Label: 1)
- **Testing**: 15 videos for evaluation
- **Note**: Dataset has class imbalance (more fake videos than real)

## 🚀 Quick Start

### 1. Setup Environment

**Option A: Use Existing Virtual Environment (Recommended)**

```bash
# Navigate to project directory
cd "/Users/mohini.gangaram/Desktop/MLPostGrad/Sem3/Deep Learning/Final Project"

# Activate existing virtual environment
source deepfake_env/bin/activate
```

**Option B: Create New Virtual Environment**

```bash
# Create virtual environment
python3 -m venv deepfake_env
source deepfake_env/bin/activate  # On macOS/Linux
# or
deepfake_env\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

**Or use the automated setup script:**

```bash
./setup_env.sh
```

### 2. Run Jupyter Notebook

```bash
# Make sure environment is activated
source deepfake_env/bin/activate

# Start Jupyter
jupyter notebook DeepFakeDetection.ipynb
```

**Important**: When the notebook opens, select the kernel:
- Click **Kernel** → **Change Kernel** → **Python (deepfake_env)**

### 3. Run Streamlit Dashboard

```bash
# Make sure environment is activated
source deepfake_env/bin/activate

# Run dashboard
streamlit run DeepFakeDetection_Dashboard.py
```

The dashboard will open in your browser automatically at `http://localhost:8501`

## 📁 Project Structure

```
Final Project/
├── DeepFakeDetection.ipynb              # Main analysis notebook
├── DeepFakeDetection_Dashboard.py        # Interactive Streamlit dashboard
├── create_notebook.py                    # Notebook creation script
├── requirements.txt                      # Python dependencies
├── setup_env.sh                          # Automated environment setup script
├── deepfake_env/                         # Virtual environment (created)
├── data/                                 # Dataset folders
│   ├── Celeb-real/                       # Real video samples
│   ├── Celeb-synthesis/                  # Fake video samples
│   └── Testing/                           # Test videos
├── dashboard_data/                       # Dashboard data files
│   ├── dataset_stats.json                # Dataset statistics
│   ├── label_distribution.csv            # Label distribution data
│   └── sample_videos.csv                 # Sample video metadata
├── evaluation_results.json               # Model evaluation results
├── *.png                                  # Visualization charts
│   ├── confusion_matrices_comparison.png
│   ├── roc_curves_comparison.png
│   ├── model_comparison_bar_chart.png
│   └── ...
└── Documentation/
    ├── README.md                          # This file
    ├── QUICK_START.md                     # Quick start guide
    ├── ENV_SETUP_GUIDE.md                 # Detailed environment setup
    ├── DATASET_SETUP.md                   # Dataset setup instructions
    ├── DATASET_LABEL_ANALYSIS.md          # Label analysis guide
    ├── NOTEBOOK_EXPLANATION.md            # Notebook walkthrough
    ├── DEPLOYMENT.md                      # Deployment guide
    ├── STREAMLIT_CLOUD_DEPLOYMENT.md      # Streamlit Cloud deployment
    ├── GITHUB_AUTH_SETUP.md               # GitHub authentication setup
    └── PUSH_INSTRUCTIONS.md               # Git push instructions
```

## 🔬 Approach

### 1. Data Loading & Exploration
- Load dataset from local folders (Celeb-Real, Celeb-Synthesis, Testing)
- Dataset source: https://drive.google.com/open?id=1iLx76wsbi9itnkxSqz9BVBl4ZvnbIazj
- Analyze dataset structure and class distribution
- Visualize sample images/videos
- Generate dataset statistics and label distribution analysis

### 2. Feature Engineering
- **Spatial Features**: Mean, std, variance, histogram features
- **Frequency Features**: FFT analysis
- **Texture Features**: LBP (Local Binary Pattern), GLCM
- **Color Features**: RGB, HSV, LAB statistics

### 3. Model Architectures
- **Simple CNN**: Lightweight baseline model for video classification
- **ResNet18**: Transfer learning with pretrained ImageNet weights, adapted for video frames
- **EfficientNet**: Efficient architecture for deployment (EfficientNet-B0)
- All models process 16 frames per video for temporal consistency

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

- **Best Model**: EfficientNet
- **Accuracy**: 92.84%
- **Precision**: 91.99%
- **Recall**: 92.84%
- **F1-Score**: 90.59%
- **ROC-AUC**: 0.8578

## 🎯 Features

- **Multiple Architectures**: Simple CNN, ResNet18, EfficientNet
- **Video Processing**: Frame extraction and preprocessing pipeline
- **Feature Engineering**: Handcrafted features (spatial, frequency, texture, color)
- **Hyperparameter Optimization**: Automated tuning with Optuna
- **Comprehensive Evaluation**: Multiple metrics and visualizations
- **Interactive Dashboard**: Streamlit app with 6 pages:
  - 🏠 **Home**: Project overview and metrics
  - 📊 **Dataset Overview**: EDA and dataset statistics
  - 🤖 **Model Architectures**: Architecture details and comparisons
  - 📈 **Training & Evaluation**: Training curves, ROC curves, confusion matrices
  - 📊 **Model Comparison**: Side-by-side performance comparison
  - 🎥 **Video Upload & Testing**: Upload videos and test with all models
- **Real-time Video Testing**: Upload videos and get predictions from all models
- **Model Explainability**: Feature importance, SHAP values, Grad-CAM
- **Comprehensive Visualizations**: Training curves, ROC curves, confusion matrices, model comparisons

## 📚 Documentation

### Streamlit Dashboard

The Streamlit dashboard (`DeepFakeDetection_Dashboard.py`) provides comprehensive explanations of:
- Dataset overview and EDA
- Feature engineering methods
- Model architectures
- Training and evaluation results
- Hyperparameter tuning process
- Model comparison and selection
- Model explainability
- Real-time video testing and prediction

### Additional Documentation Files

- `QUICK_START.md` - Quick setup and usage guide
- `ENV_SETUP_GUIDE.md` - Detailed virtual environment setup instructions
- `DATASET_SETUP.md` - Dataset download and organization guide
- `DATASET_LABEL_ANALYSIS.md` - Label distribution analysis guide
- `NOTEBOOK_EXPLANATION.md` - Complete notebook walkthrough
- `DEPLOYMENT.md` - Deployment guide for Streamlit Cloud, Heroku, Docker
- `STREAMLIT_CLOUD_DEPLOYMENT.md` - Streamlit Cloud specific deployment
- `GITHUB_AUTH_SETUP.md` - GitHub authentication for dataset access
- `PUSH_INSTRUCTIONS.md` - Git workflow and push instructions

## 🔄 Recent Updates

- ✅ Complete Streamlit dashboard with 6 interactive pages
- ✅ Video upload and real-time testing functionality
- ✅ Comprehensive model comparison visualizations
- ✅ Dataset statistics and EDA integration
- ✅ Evaluation results JSON export
- ✅ Automated environment setup script
- ✅ Complete documentation suite
- ✅ Deployment guides for multiple platforms

## 🚀 Deployment

The project is ready for deployment on:
- **Streamlit Cloud** (Recommended) - See `STREAMLIT_CLOUD_DEPLOYMENT.md`
- **Heroku** - See `DEPLOYMENT.md`
- **Docker** - See `DEPLOYMENT.md`

**GitHub Repository**: https://github.com/mohini-workday/Deepfake-Detection

## 👤 Authors

Poonam and Mohini - ML PostGrad - Deep Learning Final Project

## 📝 License

This project is part of academic research.

