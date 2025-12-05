#!/bin/bash
# Setup script for DeepFake Detection Project Virtual Environment

echo "=========================================="
echo "DeepFake Detection Project Setup"
echo "=========================================="

# Navigate to project directory
cd "$(dirname "$0")"
PROJECT_DIR=$(pwd)

echo "Project directory: $PROJECT_DIR"

# Check if virtual environment exists
if [ ! -d "deepfake_env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv deepfake_env
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source deepfake_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install additional packages for Jupyter
echo "Installing Jupyter and additional packages..."
pip install jupyter ipykernel torchcodec av

# Register kernel with Jupyter
echo "Registering Jupyter kernel..."
python -m ipykernel install --user --name=deepfake_env --display-name="Python (deepfake_env)"

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
python -c "import torch; import torchvision; import datasets; import streamlit; print('✅ All key packages imported successfully'); print(f'PyTorch: {torch.__version__}'); print(f'Torchvision: {torchvision.__version__}'); print(f'Datasets: {datasets.__version__}'); print(f'Streamlit: {streamlit.__version__}')"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  source deepfake_env/bin/activate"
echo ""
echo "To run Jupyter Notebook:"
echo "  jupyter notebook DeepFakeDetection.ipynb"
echo ""
echo "To run Streamlit Dashboard:"
echo "  streamlit run DeepFakeDetection_Dashboard.py"
echo ""

