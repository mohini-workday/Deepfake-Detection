# Streamlit Dashboard for DeepFake Detection Project
# Comprehensive dashboard with video upload and model comparison
# Author: Mohini

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import warnings
import torch
import torch.nn as nn
from torchvision import transforms, models
import timm
import cv2
import tempfile
import os
import sys
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score,
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)

warnings.filterwarnings('ignore')

# Add project root to path for model imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_data
def load_dataset_stats():
    """Load dataset statistics from saved files"""
    stats_path = PROJECT_ROOT / "dashboard_data" / "dataset_stats.json"
    label_path = PROJECT_ROOT / "dashboard_data" / "label_distribution.csv"
    
    if stats_path.exists():
        import json
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        return stats, None
    elif label_path.exists():
        label_df = pd.read_csv(label_path)
        return None, label_df
    else:
        # Fallback to hardcoded data
        return None, pd.DataFrame({
            'Label': ['Fake', 'Celeb-Real'],
            'Count': [5582, 493],
            'Percentage': [91.9, 8.1]
        })

@st.cache_data
def load_evaluation_results():
    """Load evaluation results from saved files (if available)"""
    eval_path = PROJECT_ROOT / "evaluation_results.json"
    
    if eval_path.exists():
        import json
        with open(eval_path, 'r') as f:
            results = json.load(f)
        return results
    return None

# Page configuration
st.set_page_config(
    page_title="DeepFake Detection Dashboard",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #1f77b4;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .prediction-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px solid #1f77b4;
    }
    .real-prediction {
        border-color: #2ecc71;
        background-color: #d4edda;
    }
    .fake-prediction {
        border-color: #e74c3c;
        background-color: #f8d7da;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL DEFINITIONS (Same as notebook)
# ============================================================================

class SimpleCNN(nn.Module):
    """Simple CNN model for video frame classification."""
    def __init__(self, num_classes=2, num_frames=16):
        super(SimpleCNN, self).__init__()
        self.num_frames = num_frames
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(batch_size * num_frames, channels, height, width)
        features = self.features(x)
        logits = self.classifier(features)
        logits = logits.view(batch_size, num_frames, -1)
        logits = logits.mean(dim=1)
        return logits


class ResNetVideoClassifier(nn.Module):
    """ResNet18-based model using transfer learning."""
    def __init__(self, num_classes=2, num_frames=16, pretrained=True):
        super(ResNetVideoClassifier, self).__init__()
        self.num_frames = num_frames
        
        resnet = models.resnet18(pretrained=pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        num_features = resnet.fc.in_features
        
        self.frame_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )
        
        self.video_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(batch_size * num_frames, channels, height, width)
        features = self.features(x)
        frame_features = self.frame_classifier(features)
        frame_features = frame_features.view(batch_size, num_frames, -1)
        video_features = frame_features.mean(dim=1)
        logits = self.video_classifier(video_features)
        return logits


class EfficientNetVideoClassifier(nn.Module):
    """EfficientNet-based model using transfer learning."""
    def __init__(self, num_classes=2, num_frames=16, model_name='efficientnet_b0', pretrained=True):
        super(EfficientNetVideoClassifier, self).__init__()
        self.num_frames = num_frames
        
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0,
            global_pool=''
        )
        
        num_features = self.backbone.num_features
        
        self.frame_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
        )
        
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        self.video_classifier = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        batch_size, num_frames, channels, height, width = x.size()
        x = x.view(batch_size * num_frames, channels, height, width)
        features = self.backbone.forward_features(x)
        frame_features = self.frame_extractor(features)
        frame_features = frame_features.view(batch_size, num_frames, -1)
        lstm_out, (hidden, cell) = self.lstm(frame_features)
        video_features = lstm_out[:, -1, :]
        logits = self.video_classifier(video_features)
        return logits

# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

@st.cache_resource
def load_model(model_type, model_path=None):
    """Load a trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_type == "SimpleCNN":
        model = SimpleCNN(num_classes=2, num_frames=16)
    elif model_type == "ResNet18":
        model = ResNetVideoClassifier(num_classes=2, num_frames=16, pretrained=True)
    elif model_type == "EfficientNet":
        model = EfficientNetVideoClassifier(num_classes=2, num_frames=16, model_name='efficientnet_b0', pretrained=True)
    else:
        return None
    
    model = model.to(device)
    model.eval()
    
    # Load weights if path provided
    if model_path and Path(model_path).exists():
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except:
            st.warning(f"Could not load weights from {model_path}. Using pretrained/random weights.")
    
    return model, device

# ============================================================================
# VIDEO PROCESSING FUNCTIONS
# ============================================================================

def extract_frames_from_video(video_path, num_frames=16):
    """Extract frames from video file"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < num_frames:
        frame_indices = list(range(0, total_frames))
    else:
        step = max(1, total_frames // num_frames)
        frame_indices = [i * step for i in range(num_frames)]
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frames.append(frame_pil)
        else:
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(Image.new('RGB', (224, 224), (0, 0, 0)))
    
    cap.release()
    
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else Image.new('RGB', (224, 224), (0, 0, 0)))
    
    frames = frames[:num_frames]
    return frames

def preprocess_frames(frames):
    """Preprocess frames for model input"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    processed_frames = [transform(frame) for frame in frames]
    frames_tensor = torch.stack(processed_frames)
    frames_tensor = frames_tensor.unsqueeze(0)  # Add batch dimension
    return frames_tensor

def predict_video(model, frames_tensor, device):
    """Make prediction on video frames"""
    with torch.no_grad():
        frames_tensor = frames_tensor.to(device)
        outputs = model(frames_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence, probabilities[0].cpu().numpy()

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
st.sidebar.title("📊 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "📊 Dataset Overview",
        "🤖 Model Architectures",
        "📈 Training & Evaluation",
        "📊 Model Comparison",
        "🎥 Video Upload & Testing"
    ]
)

# ============================================================================
# PAGE 1: HOME
# ============================================================================
if page == "🏠 Home":
    st.title("🎭 DeepFake Detection Dashboard")
    st.markdown("---")
    
    st.markdown("""
    <div class="info-box">
        <h3>📋 Project Overview</h3>
        <p><strong>Objective:</strong> Develop models to detect deepfake videos with high accuracy and explainability</p>
        <p><strong>Business Value:</strong> Flagging misinformation / Protecting digital identity</p>
        <p><strong>Dataset:</strong> Google Drive - Celeb-Real, Celeb-Fake, and Testing folders</p>
        <p><strong>Approach:</strong> Multiple CNN architectures + Transfer Learning + Hyperparameter Optimization</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Models Trained", "3")
        st.caption("Simple CNN, ResNet18, EfficientNet")
    
    with col2:
        st.metric("Dataset Size", "6,075 videos")
        st.caption("Celeb-Real + Celeb-Fake")
    
    with col3:
        st.metric("Test Videos", "15 videos")
        st.caption("Separate test set")
    
    st.markdown("---")
    
    st.subheader("🎯 Project Pipeline")
    st.markdown("""
    1. **Data Loading** - Load videos from local folders (Celeb-Real, Celeb-Fake, Testing)
    2. **Feature Engineering** - Extract frames and apply preprocessing
    3. **Model Development** - Implement 3 CNN architectures
    4. **Training** - Train models with data augmentation
    5. **Evaluation** - Comprehensive metrics and visualizations
    6. **Model Comparison** - Compare all 3 models
    7. **Video Testing** - Upload and test videos with all models
    """)
    
    st.subheader("📚 Key Features")
    st.markdown("""
    - **Multiple Architectures**: Simple CNN, ResNet18 (Transfer Learning), EfficientNet + LSTM
    - **Video Processing**: Frame extraction and temporal modeling
    - **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score
    - **Interactive Testing**: Upload videos and compare model predictions
    - **Visualizations**: Training curves, confusion matrices, model comparisons
    """)

# ============================================================================
# PAGE 2: DATASET OVERVIEW
# ============================================================================
elif page == "📊 Dataset Overview":
    st.header("📊 Dataset Overview & Exploratory Data Analysis")
    
    # Load data
    stats, label_data = load_dataset_stats()
    
    if stats is None and label_data is None:
        st.warning("⚠️ Dataset statistics not found. Please run Cell 10 in the notebook to generate statistics.")
        label_data = pd.DataFrame({
            'Label': ['Fake', 'Celeb-Real'],
            'Count': [5582, 493],
            'Percentage': [91.9, 8.1]
        })
    elif label_data is None:
        # Create label_data from stats
        label_data = pd.DataFrame({
            'Label': list(stats['label_distribution'].keys()),
            'Count': list(stats['label_distribution'].values())
        })
        label_data['Percentage'] = (label_data['Count'] / label_data['Count'].sum() * 100).round(2)
    
    st.subheader("Dataset Information")
    st.markdown("""
    **Source**: Google Drive - Local folders
    
    - **Celeb-Real**: Real/Original videos (Label: 0, "Celeb-Real")
    - **Celeb-Fake**: Fake/Manipulated videos (Label: 1, "Fake") 
    - **Testing**: Test videos for evaluation
    """)
    
    # Display statistics
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Videos", stats.get('total_videos', 'N/A'))
        with col2:
            st.metric("Real Videos", stats.get('real_videos', 'N/A'))
        with col3:
            st.metric("Fake Videos", stats.get('fake_videos', 'N/A'))
        with col4:
            if stats.get('total_videos'):
                imbalance_ratio = stats.get('fake_videos', 0) / stats.get('real_videos', 1)
                st.metric("Imbalance Ratio", f"{imbalance_ratio:.1f}:1")
    
    st.markdown("---")
    
    # CHART 1: Label Distribution
    st.subheader("📈 Chart 1: Label Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            label_data,
            x='Label',
            y='Count',
            title='Label Distribution (Count)',
            color='Label',
            color_discrete_map={'Celeb-Real': '#2ecc71', 'Fake': '#e74c3c'},
            text='Count'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.pie(
            label_data,
            values='Count',
            names='Label',
            title='Label Distribution (Percentage)',
            color='Label',
            color_discrete_map={'Celeb-Real': '#2ecc71', 'Fake': '#e74c3c'},
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    # CHART 2: Folder Distribution
    if stats and 'folders' in stats:
        st.subheader("📁 Chart 2: Distribution by Source Folder")
        
        folder_data = pd.DataFrame({
            'Folder': list(stats['folders'].keys()),
            'Count': list(stats['folders'].values())
        })
        
        fig = px.bar(
            folder_data,
            x='Folder',
            y='Count',
            title='Videos by Source Folder',
            color='Folder',
            text='Count'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(height=400, xaxis_title='Folder', yaxis_title='Number of Videos')
        st.plotly_chart(fig, use_container_width=True)
    
    # CHART 3: Video Format Distribution
    if stats and 'video_formats' in stats:
        st.subheader("🎬 Chart 3: Video Format Distribution")
        
        format_data = pd.DataFrame({
            'Format': list(stats['video_formats'].keys()),
            'Count': list(stats['video_formats'].values())
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                format_data,
                x='Format',
                y='Count',
                title='Videos by File Format',
                color='Format',
                text='Count'
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(
                format_data,
                values='Count',
                names='Format',
                title='Format Distribution',
                hole=0.3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
    
    # CHART 4: Class Imbalance Visualization
    st.subheader("⚖️ Chart 4: Class Imbalance Analysis")
    
    if len(label_data) == 2:
        imbalance_data = pd.DataFrame({
            'Metric': ['Real Videos', 'Fake Videos'],
            'Count': [label_data[label_data['Label'] == 'Celeb-Real']['Count'].iloc[0] if len(label_data[label_data['Label'] == 'Celeb-Real']) > 0 else 0,
                      label_data[label_data['Label'] == 'Fake']['Count'].iloc[0] if len(label_data[label_data['Label'] == 'Fake']) > 0 else 0]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=imbalance_data['Metric'],
            y=imbalance_data['Count'],
            marker_color=['#2ecc71', '#e74c3c'],
            text=imbalance_data['Count'],
            textposition='outside'
        ))
        fig.update_layout(
            title='Class Imbalance Visualization',
            xaxis_title='Class',
            yaxis_title='Number of Videos',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Imbalance ratio
        if len(imbalance_data) == 2:
            ratio = imbalance_data.iloc[1]['Count'] / imbalance_data.iloc[0]['Count'] if imbalance_data.iloc[0]['Count'] > 0 else 0
            st.info(f"⚠️ **Class Imbalance**: Fake videos are {ratio:.1f}x more than Real videos. Consider using class weights or data augmentation.")
    
    # CHART 5: Summary Statistics Table
    st.subheader("📊 Chart 5: Dataset Summary Statistics")
    
    summary_stats = []
    if stats:
        summary_stats.append({'Metric': 'Total Videos', 'Value': stats.get('total_videos', 'N/A')})
        summary_stats.append({'Metric': 'Real Videos', 'Value': stats.get('real_videos', 'N/A')})
        summary_stats.append({'Metric': 'Fake Videos', 'Value': stats.get('fake_videos', 'N/A')})
        if stats.get('total_videos'):
            summary_stats.append({'Metric': 'Real Percentage', 'Value': f"{(stats.get('real_videos', 0) / stats.get('total_videos', 1) * 100):.2f}%"})
            summary_stats.append({'Metric': 'Fake Percentage', 'Value': f"{(stats.get('fake_videos', 0) / stats.get('total_videos', 1) * 100):.2f}%"})
    
    if summary_stats:
        summary_df = pd.DataFrame(summary_stats)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("""
    <div class="info-box">
        <h4>📝 Dataset Insights</h4>
        <ul>
            <li><strong>Class Imbalance:</strong> Dataset has significantly more Fake videos than Real videos</li>
            <li><strong>Data Split:</strong> 80% Training, 20% Validation</li>
            <li><strong>Preprocessing:</strong> Videos converted to 16 frames, resized to 224x224</li>
            <li><strong>Augmentation:</strong> Random horizontal flip, rotation, color jitter</li>
            <li><strong>Recommendation:</strong> Use class weights or oversampling to handle imbalance</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE 3: MODEL ARCHITECTURES
# ============================================================================
elif page == "🤖 Model Architectures":
    st.header("🤖 Model Architectures")
    
    model_tabs = st.tabs(["Simple CNN", "ResNet18", "EfficientNet"])
    
    with model_tabs[0]:
        st.subheader("1. Simple CNN Architecture")
        st.markdown("""
        <div class="model-card">
            <h4>Architecture Details</h4>
            <ul>
                <li><strong>Input:</strong> 16 frames of 224x224x3 RGB images</li>
                <li><strong>Convolutional Blocks:</strong> 4 blocks (32→64→128→256 filters)</li>
                <li><strong>Regularization:</strong> Batch Normalization, Dropout (0.5)</li>
                <li><strong>Classifier:</strong> 2 fully connected layers (256→128→2)</li>
                <li><strong>Total Parameters:</strong> ~422K</li>
                <li><strong>Frame Aggregation:</strong> Average pooling across frames</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with model_tabs[1]:
        st.subheader("2. ResNet18 (Transfer Learning)")
        st.markdown("""
        <div class="model-card">
            <h4>Architecture Details</h4>
            <ul>
                <li><strong>Base Model:</strong> ResNet18 pretrained on ImageNet</li>
                <li><strong>Transfer Learning:</strong> Fine-tuned for deepfake detection</li>
                <li><strong>Frame Features:</strong> 512→256 dimensions</li>
                <li><strong>Video Classifier:</strong> 256→128→2</li>
                <li><strong>Total Parameters:</strong> ~11.6M</li>
                <li><strong>Frame Aggregation:</strong> Average pooling</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with model_tabs[2]:
        st.subheader("3. EfficientNet (Transfer Learning + LSTM)")
        st.markdown("""
        <div class="model-card">
            <h4>Architecture Details</h4>
            <ul>
                <li><strong>Base Model:</strong> EfficientNet-B0 pretrained on ImageNet</li>
                <li><strong>Temporal Modeling:</strong> Bidirectional LSTM (2 layers)</li>
                <li><strong>Frame Features:</strong> 512→256 dimensions</li>
                <li><strong>LSTM:</strong> 256→128 (bidirectional = 256)</li>
                <li><strong>Video Classifier:</strong> 256→128→2</li>
                <li><strong>Total Parameters:</strong> ~5.6M</li>
                <li><strong>Frame Aggregation:</strong> LSTM temporal modeling</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("📊 Architecture Comparison")
    
    comparison_data = pd.DataFrame({
        'Model': ['Simple CNN', 'ResNet18', 'EfficientNet'],
        'Parameters (M)': [0.42, 11.6, 5.6],
        'Temporal Modeling': ['Average', 'Average', 'LSTM'],
        'Transfer Learning': ['No', 'Yes', 'Yes']
    })
    
    st.dataframe(comparison_data, use_container_width=True)

# ============================================================================
# PAGE 4: TRAINING & EVALUATION
# ============================================================================
elif page == "📈 Training & Evaluation":
    st.header("📈 Training & Evaluation")
    
    # Try to load evaluation results
    eval_results = load_evaluation_results()
    
    if eval_results is None:
        st.info("💡 **Note**: To view ROC curves and confusion matrices, please run Cell 12 in the notebook to generate evaluation results. The results will be saved and displayed here automatically.")
        
        st.subheader("Expected Training Metrics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        metrics = {
            'Accuracy': '91-93%',
            'Precision': '91-93%',
            'Recall': '91-93%',
            'F1-Score': '91-93%',
            'ROC-AUC': '95-98%'
        }
        
        for col, (metric, value) in zip([col1, col2, col3, col4, col5], metrics.items()):
            with col:
                st.metric(metric, value)
        
        st.subheader("Training Configuration")
        st.markdown("""
        - **Epochs**: 2-10 (configurable)
        - **Batch Size**: 4
        - **Learning Rate**: 0.001 (Simple CNN), 0.0001 (Transfer Learning models)
        - **Optimizer**: Adam with weight decay (1e-4)
        - **Scheduler**: ReduceLROnPlateau
        - **Loss Function**: CrossEntropyLoss
        """)
    else:
        # Display actual evaluation results
        st.success("✅ Evaluation results loaded successfully!")
        
        # ============================================================================
        # METRICS SUMMARY
        # ============================================================================
        st.subheader("📊 Model Performance Metrics")
        
        metrics_data = []
        for model_name, results in eval_results.items():
            metrics_data.append({
                'Model': model_name,
                'Accuracy (%)': f"{results['accuracy']*100:.2f}",
                'Precision (%)': f"{results['precision']*100:.2f}",
                'Recall (%)': f"{results['recall']*100:.2f}",
                'F1-Score (%)': f"{results['f1']*100:.2f}",
                'ROC-AUC': f"{results['roc_auc']:.4f}"
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # ============================================================================
        # ROC CURVES
        # ============================================================================
        st.subheader("📈 ROC Curves - Model Comparison")
        
        # Check if we have actual data (not empty arrays)
        has_data = False
        for model_name, results in eval_results.items():
            if (len(results.get('true_labels', [])) > 0 and 
                len(results.get('probabilities', [])) > 0):
                has_data = True
                break
        
        if not has_data:
            st.warning("⚠️ **Note**: ROC curves require actual prediction data. The evaluation_results.json file exists but contains empty arrays. Please run Cell 12 in the notebook to generate actual evaluation data with predictions and probabilities.")
            st.info("💡 **To generate the data**: Run Cell 12 in DeepFakeDetection.ipynb after training models. This will populate the arrays with actual predictions.")
        else:
            # Create ROC curves using Plotly
            fig_roc = go.Figure()
            
            colors = ['#3498db', '#e74c3c', '#2ecc71']  # Blue, Red, Green
            model_names_list = list(eval_results.keys())
            
            for idx, (model_name, results) in enumerate(eval_results.items()):
                # Convert to numpy arrays if they're lists
                true_labels = np.array(results['true_labels'])
                probabilities = np.array(results['probabilities'])
                
                # Skip if arrays are empty
                if len(true_labels) == 0 or len(probabilities) == 0:
                    st.warning(f"⚠️ {model_name}: No data available for ROC curve")
                    continue
                
                # Calculate ROC curve
                try:
                    fpr, tpr, thresholds = roc_curve(true_labels, probabilities)
                    roc_auc = results['roc_auc']
                    
                    # Add ROC curve trace
                    fig_roc.add_trace(go.Scatter(
                        x=fpr,
                        y=tpr,
                        mode='lines',
                        name=f"{model_name} (AUC = {roc_auc:.4f})",
                        line=dict(color=colors[idx % len(colors)], width=3),
                        hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
                    ))
                except Exception as e:
                    st.error(f"Error generating ROC curve for {model_name}: {str(e)}")
                    continue
        
            # Add diagonal line (random classifier) only if we have curves
            if len(fig_roc.data) > 0:
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode='lines',
                    name='Random Classifier (AUC = 0.5000)',
                    line=dict(color='black', width=2, dash='dash'),
                    hovertemplate='Random Classifier<extra></extra>'
                ))
                
                # Update layout
                fig_roc.update_layout(
                    title='ROC Curves - Model Comparison',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    width=900,
                    height=700,
                    hovermode='closest',
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    ),
                    template='plotly_white',
                    xaxis=dict(range=[0, 1]),
                    yaxis=dict(range=[0, 1.05])
                )
                
                st.plotly_chart(fig_roc, use_container_width=True)
        
        st.markdown("""
        **Interpretation:**
        - **ROC Curve**: Shows the trade-off between True Positive Rate (TPR) and False Positive Rate (FPR)
        - **AUC (Area Under Curve)**: Higher AUC indicates better model performance
        - **Diagonal Line**: Represents a random classifier (AUC = 0.5)
        - **Best Model**: The curve closest to the top-left corner (highest AUC) performs best
        """)
        
        st.markdown("---")
        
        # ============================================================================
        # CONFUSION MATRICES
        # ============================================================================
        st.subheader("📊 Confusion Matrices - Model Comparison")
        
        # Check if we have data for confusion matrices
        has_cm_data = False
        for model_name, results in eval_results.items():
            if (len(results.get('true_labels', [])) > 0 and 
                len(results.get('predictions', [])) > 0):
                has_cm_data = True
                break
        
        if not has_cm_data:
            st.warning("⚠️ **Note**: Confusion matrices require actual prediction data. Please run Cell 12 in the notebook to generate actual evaluation data.")
        else:
            class_names = ['Celeb-Real', 'Fake']
            
            # Create tabs for each model's confusion matrix
            cm_tabs = st.tabs([f"{name} Confusion Matrix" for name in eval_results.keys()])
            
            for tab_idx, (model_name, results) in enumerate(eval_results.items()):
                with cm_tabs[tab_idx]:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Convert to numpy arrays if they're lists
                        true_labels = np.array(results['true_labels'])
                        predictions = np.array(results['predictions'])
                        
                        # Skip if arrays are empty
                        if len(true_labels) == 0 or len(predictions) == 0:
                            st.warning(f"⚠️ {model_name}: No data available for confusion matrix")
                            continue
                        
                        # Calculate confusion matrix
                        try:
                            cm = confusion_matrix(true_labels, predictions)
                            
                            # Normalize confusion matrix
                            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
                            
                            # Create heatmap using Plotly
                            fig_cm = go.Figure(data=go.Heatmap(
                                z=cm,
                                x=class_names,
                                y=class_names,
                                colorscale='Blues',
                                text=[[f"{cm[i, j]}<br>({cm_normalized[i, j]:.1f}%)" 
                                       for j in range(len(class_names))] 
                                      for i in range(len(class_names))],
                                texttemplate="%{text}",
                                textfont={"size": 14, "color": "white"},
                                colorbar=dict(title="Count")
                            ))
                            
                            fig_cm.update_layout(
                                title=f'{model_name} - Confusion Matrix',
                                xaxis_title='Predicted Label',
                                yaxis_title='True Label',
                                width=600,
                                height=500,
                                template='plotly_white'
                            )
                            
                            st.plotly_chart(fig_cm, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error generating confusion matrix for {model_name}: {str(e)}")
                    
                    with col2:
                        st.markdown("### Metrics")
                        st.metric("Accuracy", f"{results['accuracy']*100:.2f}%")
                        st.metric("Precision", f"{results['precision']*100:.2f}%")
                        st.metric("Recall", f"{results['recall']*100:.2f}%")
                        st.metric("F1-Score", f"{results['f1']*100:.2f}%")
                        st.metric("ROC-AUC", f"{results['roc_auc']:.4f}")
                        
                        st.markdown("### Confusion Matrix")
                        st.markdown("""
                        - **True Negative (TN)**: Correctly predicted Real
                        - **False Positive (FP)**: Incorrectly predicted Fake (Real labeled as Fake)
                        - **False Negative (FN)**: Incorrectly predicted Real (Fake labeled as Real)
                        - **True Positive (TP)**: Correctly predicted Fake
                        """)
        
        st.markdown("---")
        
        # ============================================================================
        # NORMALIZED CONFUSION MATRICES
        # ============================================================================
        if has_cm_data:
            st.subheader("📊 Normalized Confusion Matrices (Percentages)")
            
            # Create normalized confusion matrices
            fig_norm, axes = plt.subplots(1, len(eval_results), figsize=(6*len(eval_results), 5))
            
            if len(eval_results) == 1:
                axes = [axes]
            
            for idx, (model_name, results) in enumerate(eval_results.items()):
                true_labels = np.array(results['true_labels'])
                predictions = np.array(results['predictions'])
                
                if len(true_labels) == 0 or len(predictions) == 0:
                    continue
                
                try:
                    cm = confusion_matrix(true_labels, predictions)
                    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
                    
                    sns.heatmap(
                        cm_normalized,
                        annot=True,
                        fmt='.2f',
                        cmap='Oranges',
                        xticklabels=class_names,
                        yticklabels=class_names,
                        ax=axes[idx],
                        cbar_kws={'label': 'Percentage (%)'}
                    )
                    
                    axes[idx].set_title(f'{model_name}\nNormalized Confusion Matrix', 
                                       fontsize=12, fontweight='bold')
                    axes[idx].set_xlabel('Predicted Label', fontsize=11)
                    axes[idx].set_ylabel('True Label', fontsize=11)
                except Exception as e:
                    st.error(f"Error generating normalized confusion matrix for {model_name}: {str(e)}")
            
            plt.tight_layout()
            st.pyplot(fig_norm)
        
        st.markdown("---")
        
        # ============================================================================
        # CLASSIFICATION REPORTS
        # ============================================================================
        if has_cm_data:
            st.subheader("📋 Detailed Classification Reports")
            
            for model_name, results in eval_results.items():
                true_labels = np.array(results['true_labels'])
                predictions = np.array(results['predictions'])
                
                if len(true_labels) == 0 or len(predictions) == 0:
                    continue
                
                with st.expander(f"{model_name} - Classification Report"):
                    try:
                        report = classification_report(
                            true_labels,
                            predictions,
                            target_names=class_names,
                            digits=4,
                            output_dict=True
                        )
                        
                        # Convert to DataFrame for better display
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error generating classification report for {model_name}: {str(e)}")
        
        st.markdown("---")
        
        st.subheader("Training Configuration")
        st.markdown("""
        - **Epochs**: 2-10 (configurable)
    - **Batch Size**: 4
    - **Learning Rate**: 0.001 (Simple CNN), 0.0001 (Transfer Learning models)
    - **Optimizer**: Adam with weight decay (1e-4)
    - **Scheduler**: ReduceLROnPlateau
    - **Loss Function**: CrossEntropyLoss
    """)

# ============================================================================
# PAGE 5: MODEL COMPARISON
# ============================================================================
elif page == "📊 Model Comparison":
    st.header("📊 Model Comparison")
    
    st.subheader("Performance Metrics Comparison")
    
    # Simulated comparison data (update with actual results after training)
    comparison_metrics = pd.DataFrame({
        'Model': ['Simple CNN', 'ResNet18', 'EfficientNet'],
        'Parameters': ['422K', '11.6M', '5.6M'],
        'Accuracy': ['TBD', 'TBD', 'TBD'],
        'Precision': ['TBD', 'TBD', 'TBD'],
        'Recall': ['TBD', 'TBD', 'TBD'],
        'F1-Score': ['TBD', 'TBD', 'TBD']
    })
    
    st.dataframe(comparison_metrics, use_container_width=True)
    
    st.info("💡 **Note**: Metrics will be updated after training. Use the 'Video Upload & Testing' page to test models on your videos.")

# ============================================================================
# PAGE 6: VIDEO UPLOAD & TESTING
# ============================================================================
elif page == "🎥 Video Upload & Testing":
    st.header("🎥 Video Upload & Testing")
    st.markdown("Upload a video to test with all three models and compare predictions.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file to test with all models"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_path = tmp_file.name
        
        st.success(f"✅ Video uploaded: {uploaded_file.name}")
        
        # Display video info
        col1, col2 = st.columns(2)
        with col1:
            st.video(uploaded_file)
        
        with col2:
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
            st.info(f"**File Size**: {file_size:.2f} MB")
            st.info(f"**File Type**: {uploaded_file.type}")
        
        st.markdown("---")
        
        # Process video
        with st.spinner("🔄 Processing video and extracting frames..."):
            frames = extract_frames_from_video(tmp_path, num_frames=16)
            
            if frames is None:
                st.error("❌ Could not process video. Please check the file format.")
                os.unlink(tmp_path)
                st.stop()
            
            # Display sample frames
            st.subheader("📸 Extracted Frames (Sample)")
            frame_cols = st.columns(4)
            for i, col in enumerate(frame_cols[:4]):
                with col:
                    st.image(frames[i * 4], caption=f"Frame {i * 4 + 1}", use_container_width=True)
            
            # Preprocess frames
            frames_tensor = preprocess_frames(frames)
        
        st.markdown("---")
        
        # Model predictions
        st.subheader("🤖 Model Predictions")
        
        models_to_test = {
            "Simple CNN": "SimpleCNN",
            "ResNet18": "ResNet18",
            "EfficientNet": "EfficientNet"
        }
        
        predictions = {}
        
        for model_name, model_type in models_to_test.items():
            with st.expander(f"🔍 {model_name} Prediction", expanded=True):
                try:
                    # Load model
                    model, device = load_model(model_type)
                    
                    # Make prediction
                    with st.spinner(f"Running {model_name}..."):
                        pred_class, confidence, probabilities = predict_video(model, frames_tensor, device)
                    
                    predictions[model_name] = {
                        'class': pred_class,
                        'confidence': confidence,
                        'probabilities': probabilities,
                        'label': 'Celeb-Real' if pred_class == 0 else 'Fake'
                    }
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        label_color = "#2ecc71" if pred_class == 0 else "#e74c3c"
                        label_text = "🎭 **REAL** (Celeb-Real)" if pred_class == 0 else "🎭 **FAKE**"
                        st.markdown(f"""
                        <div class="prediction-box {'real-prediction' if pred_class == 0 else 'fake-prediction'}">
                            <h3>{label_text}</h3>
                            <p><strong>Confidence:</strong> {confidence*100:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)
    
                    with col2:
                        # Probability bar chart
                        prob_df = pd.DataFrame({
                            'Class': ['Celeb-Real', 'Fake'],
                            'Probability': [probabilities[0], probabilities[1]]
                        })
                        
                        fig = px.bar(
                            prob_df,
                            x='Class',
                            y='Probability',
                            title=f'{model_name} Probabilities',
                            color='Class',
                            color_discrete_map={'Celeb-Real': '#2ecc71', 'Fake': '#e74c3c'},
                            text=[f'{p*100:.1f}%' for p in probabilities]
                        )
                        fig.update_layout(height=300, yaxis_title='Probability', yaxis_range=[0, 1])
                        fig.update_traces(textposition='outside')
                        st.plotly_chart(fig, use_container_width=True)
    
                except Exception as e:
                    st.error(f"Error with {model_name}: {str(e)}")
                    predictions[model_name] = None
        
        st.markdown("---")
        
        # Model Comparison
        if all(p is not None for p in predictions.values()):
            st.subheader("📊 Model Comparison")
            
            # Comparison table
            comparison_data = []
            for model_name, pred in predictions.items():
                comparison_data.append({
                    'Model': model_name,
                    'Prediction': pred['label'],
                    'Confidence': f"{pred['confidence']*100:.2f}%",
                    'Real Prob': f"{pred['probabilities'][0]*100:.2f}%",
                    'Fake Prob': f"{pred['probabilities'][1]*100:.2f}%"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Comparison visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # Confidence comparison
                conf_data = pd.DataFrame({
                    'Model': list(predictions.keys()),
                    'Confidence': [p['confidence']*100 for p in predictions.values()]
                })
                
                fig = px.bar(
                    conf_data,
                    x='Model',
                    y='Confidence',
                    title='Model Confidence Comparison',
                    color='Confidence',
                    color_continuous_scale='RdYlGn',
                    text=[f'{c:.1f}%' for c in conf_data['Confidence']]
                )
                fig.update_layout(height=400, yaxis_title='Confidence (%)', yaxis_range=[0, 100])
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Agreement visualization
                predictions_list = [p['label'] for p in predictions.values()]
                agreement = len(set(predictions_list)) == 1
                
                if agreement:
                    st.success(f"✅ **All models agree**: {predictions_list[0]}")
                else:
                    st.warning("⚠️ **Models disagree** on prediction")
                    st.write("Predictions:")
                    for model, pred in predictions.items():
                        st.write(f"- {model}: {pred['label']} ({pred['confidence']*100:.1f}%)")
            
            # Consensus prediction
            st.markdown("---")
            st.subheader("🎯 Consensus Prediction")
            
            # Calculate average probabilities
            avg_probs = np.mean([p['probabilities'] for p in predictions.values()], axis=0)
            consensus_class = np.argmax(avg_probs)
            consensus_label = 'Celeb-Real' if consensus_class == 0 else 'Fake'
            consensus_confidence = avg_probs[consensus_class]
            
            st.markdown(f"""
            <div class="prediction-box {'real-prediction' if consensus_class == 0 else 'fake-prediction'}">
                <h2>Consensus: {consensus_label}</h2>
                <p><strong>Average Confidence:</strong> {consensus_confidence*100:.2f}%</p>
                <p><strong>Agreement:</strong> {sum(1 for p in predictions.values() if p['class'] == consensus_class)}/3 models</p>
    </div>
    """, unsafe_allow_html=True)
    
        # Cleanup
        os.unlink(tmp_path)
    
    else:
        st.info("👆 Please upload a video file to begin testing.")
    st.markdown("""
        ### 📝 Instructions:
        1. Upload a video file (MP4, AVI, MOV, MKV)
        2. Wait for frame extraction
        3. View predictions from all 3 models
        4. Compare results and see consensus prediction
        """)

# ============================================================================
# FOOTER
# ============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📚 About
**DeepFake Detection Dashboard**

Comprehensive analysis and video testing for deepfake detection models

**Author**: Mohini  
**Project**: ML PostGrad - Deep Learning Final Project
""")
