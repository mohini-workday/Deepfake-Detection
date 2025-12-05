# Virtual Environment Setup Guide

## ✅ Virtual Environment Created

A new virtual environment `deepfake_env` has been created for the DeepFake Detection project.

## 📦 Installed Packages

All required packages from `requirements.txt` have been installed:
- **Deep Learning**: PyTorch, Torchvision, Torchaudio, TIMM
- **Data Processing**: NumPy, Pandas, OpenCV, Pillow, scikit-image
- **Machine Learning**: scikit-learn, Optuna
- **HuggingFace**: datasets, transformers
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Dashboard**: Streamlit
- **Utilities**: tqdm, joblib
- **Jupyter**: Jupyter Notebook, IPython Kernel
- **Video Processing**: torchcodec, av (PyAV)

## 🚀 How to Use

### 1. Activate the Virtual Environment

```bash
cd "/Users/mohini.gangaram/Desktop/MLPostGrad/Sem3/Deep Learning/Final Project"
source deepfake_env/bin/activate
```

You'll see `(deepfake_env)` in your terminal prompt when activated.

### 2. Run Jupyter Notebook

```bash
# Make sure environment is activated
source deepfake_env/bin/activate

# Start Jupyter Notebook
jupyter notebook DeepFakeDetection.ipynb
```

**Important**: When opening the notebook, select the kernel:
- Click on "Kernel" → "Change Kernel" → "Python (deepfake_env)"

### 3. Run Streamlit Dashboard

```bash
# Make sure environment is activated
source deepfake_env/bin/activate

# Run Streamlit dashboard
streamlit run DeepFakeDetection_Dashboard.py
```

### 4. Deactivate Environment

When you're done:
```bash
deactivate
```

## 🔧 Quick Setup (Re-run if needed)

If you need to recreate the environment or install packages again:

```bash
cd "/Users/mohini.gangaram/Desktop/MLPostGrad/Sem3/Deep Learning/Final Project"
./setup_env.sh
```

Or manually:
```bash
source deepfake_env/bin/activate
pip install -r requirements.txt
pip install jupyter ipykernel torchcodec av
python -m ipykernel install --user --name=deepfake_env --display-name="Python (deepfake_env)"
```

## 📋 Environment Details

- **Python Version**: 3.14.0
- **Environment Name**: `deepfake_env`
- **Location**: `/Users/mohini.gangaram/Desktop/MLPostGrad/Sem3/Deep Learning/Final Project/deepfake_env`
- **Jupyter Kernel**: `Python (deepfake_env)`

## ✅ Verification

To verify all packages are installed correctly:

```bash
source deepfake_env/bin/activate
python -c "import torch; import torchvision; import datasets; import streamlit; print('✅ All packages working!')"
```

## 🐛 Troubleshooting

### If Jupyter doesn't see the kernel:
```bash
source deepfake_env/bin/activate
python -m ipykernel install --user --name=deepfake_env --display-name="Python (deepfake_env)"
```

### If packages are missing:
```bash
source deepfake_env/bin/activate
pip install -r requirements.txt
```

### If you get import errors:
1. Make sure the environment is activated (`source deepfake_env/bin/activate`)
2. Check that you selected the correct kernel in Jupyter
3. Restart the Jupyter kernel after installing new packages

## 📝 Notes

- The virtual environment is isolated from your system Python
- All project dependencies are contained within `deepfake_env/`
- The environment can be deleted and recreated anytime using `setup_env.sh`
- Don't commit the `deepfake_env/` folder to git (it's in `.gitignore`)

