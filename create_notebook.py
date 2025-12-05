import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "Project : Develop model to detect deepfake video with Highest accuracy (possible by us) which has explainability . Will try to create few model to generate comparisons for comparison and then picking one final one as our \"final model\".\n",
                "\n",
                "Business Value: Flagging misinformation/ protecting digital identity"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Deep Fake Detection Project - Complete Pipeline\n",
                "!pip install optuna torchcodec\n",
                "\n",
                "import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns\n",
                "import torch, torch.nn as nn, torch.optim as optim\n",
                "from torch.utils.data import Dataset, DataLoader, random_split\n",
                "from torchvision import transforms, models\n",
                "import timm, cv2\n",
                "from PIL import Image\n",
                "from skimage.feature import local_binary_pattern\n",
                "from datasets import load_dataset\n",
                "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score\n",
                "import optuna\n",
                "from tqdm.auto import tqdm\n",
                "import warnings\n",
                "warnings.filterwarnings('ignore')\n",
                "\n",
                "SEED = 42\n",
                "np.random.seed(SEED)\n",
                "torch.manual_seed(SEED)\n",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "print(f\"Using device: {device}\")"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.8.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open('DeepFakeDetection.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook created successfully!")
