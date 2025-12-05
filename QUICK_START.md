# Quick Start Guide - DeepFake Detection Project

## 🚀 Quick Setup (Already Done!)

The virtual environment `deepfake_env` has been created and all packages are installed.

## 📝 How to Use

### 1. Activate Virtual Environment

```bash
cd "/Users/mohini.gangaram/Desktop/MLPostGrad/Sem3/Deep Learning/Final Project"
source deepfake_env/bin/activate
```

You'll see `(deepfake_env)` in your terminal prompt.

### 2. Run Jupyter Notebook

```bash
# Make sure environment is activated first
source deepfake_env/bin/activate

# Start Jupyter
jupyter notebook DeepFakeDetection.ipynb
```

**Important**: When the notebook opens, select the kernel:
- Click **Kernel** → **Change Kernel** → **Python (deepfake_env)**

### 3. Run Streamlit Dashboard

```bash
# Make sure environment is activated first
source deepfake_env/bin/activate

# Run dashboard
streamlit run DeepFakeDetection_Dashboard.py
```

The dashboard will open in your browser automatically.

### 4. Deactivate Environment

When done:
```bash
deactivate
```

## ✅ Verification

To verify everything is working:

```bash
source deepfake_env/bin/activate
python -c "import torch; import datasets; import streamlit; print('✅ All packages working!')"
```

## 📦 Installed Packages

- **PyTorch 2.9.1** - Deep learning framework
- **Torchvision 0.24.1** - Computer vision utilities
- **HuggingFace Datasets 4.4.1** - Dataset loading
- **Streamlit 1.52.0** - Dashboard framework
- **Jupyter** - Notebook environment
- **Optuna 4.6.0** - Hyperparameter optimization
- **TIMM 1.0.22** - Model architectures
- **OpenCV** - Image/video processing
- And many more...

## 🎯 Next Steps

1. **Run the notebook** to analyze your dataset labels
2. **Check label distribution** - See if you have both Original and Manipulated videos
3. **Complete the notebook** - Add remaining code cells from Colab
4. **Run the dashboard** - Explore models and visualizations

## 📚 Documentation

- `README.md` - Project overview
- `ENV_SETUP_GUIDE.md` - Detailed environment setup guide
- `DATASET_LABEL_ANALYSIS.md` - Label analysis guide
- `setup_env.sh` - Automated setup script

## 🐛 Troubleshooting

**Problem**: Jupyter doesn't see the kernel
**Solution**: 
```bash
source deepfake_env/bin/activate
python -m ipykernel install --user --name=deepfake_env --display-name="Python (deepfake_env)"
```

**Problem**: Import errors in notebook
**Solution**: Make sure you selected the correct kernel: **Python (deepfake_env)**

**Problem**: Packages missing
**Solution**: 
```bash
source deepfake_env/bin/activate
pip install -r requirements.txt
```

