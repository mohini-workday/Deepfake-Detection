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

warnings.filterwarnings('ignore')

# Add project root to path for model imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

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
    st.header("📊 Dataset Overview & EDA")
    
    st.subheader("Dataset Information")
    st.markdown("""
    **Source**: Google Drive - Local folders
    
    - **Celeb-Real**: Real/Original videos (Label: 0, "Celeb-Real")
    - **Celeb-Fake**: Fake/Manipulated videos (Label: 1, "Fake") 
    - **Testing**: Test videos for evaluation
    
    - **Total Training Samples**: 6,075 videos
    - **Format**: MP4 video files
    - **Classes**: 
      - 0: Real/Original (Celeb-Real)
      - 1: Fake/Manipulated (Celeb-Fake)
    """)
    
    st.subheader("📈 Label Distribution")
    
    # Actual data from your dataset
    label_data = pd.DataFrame({
        'Label': ['Fake', 'Celeb-Real'],
        'Count': [5582, 493],
        'Percentage': [91.9, 8.1]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            label_data,
            x='Label',
            y='Count',
            title='Label Distribution (Count)',
            color='Label',
            color_discrete_map={'Celeb-Real': '#2ecc71', 'Fake': '#e74c3c'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.pie(
            label_data,
            values='Count',
            names='Label',
            title='Label Distribution (Percentage)',
            color='Label',
            color_discrete_map={'Celeb-Real': '#2ecc71', 'Fake': '#e74c3c'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="info-box">
        <h4>📝 Dataset Insights</h4>
        <ul>
            <li><strong>Class Imbalance:</strong> Dataset has more Fake videos (91.9%) than Real (8.1%)</li>
            <li><strong>Data Split:</strong> 80% Training, 20% Validation</li>
            <li><strong>Preprocessing:</strong> Videos converted to 16 frames, resized to 224x224</li>
            <li><strong>Augmentation:</strong> Random horizontal flip, rotation, color jitter</li>
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
    
    st.info("💡 **Note**: Training results will be displayed here after models are trained. You can load saved model checkpoints to view training history.")
    
    st.subheader("Expected Training Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    metrics = {
        'Accuracy': '95-100%',
        'Precision': '94-100%',
        'Recall': '96-100%',
        'F1-Score': '95-100%',
        'ROC-AUC': '98-100%'
    }
    
    for col, (metric, value) in zip([col1, col2, col3, col4, col5], metrics.items()):
        with col:
            st.metric(metric, value)
    
    st.subheader("Training Configuration")
    st.markdown("""
    - **Epochs**: 10
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
