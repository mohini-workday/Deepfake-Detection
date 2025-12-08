# Deployment Guide for Streamlit Dashboard

## ✅ Pre-Deployment Checklist

### Files Required for Deployment

- ✅ `DeepFakeDetection_Dashboard.py` - Main Streamlit application
- ✅ `requirements.txt` - All Python dependencies
- ✅ `README.md` - Project documentation
- ✅ `.gitignore` - Git ignore rules

### Dependencies Status

All required packages are in `requirements.txt`:
- ✅ Streamlit (1.25.0+)
- ✅ PyTorch (2.0.0+)
- ✅ Torchvision (0.15.0+)
- ✅ TIMM (0.9.0+)
- ✅ OpenCV (4.8.0+)
- ✅ Plotly (5.14.0+)
- ✅ Pandas, NumPy, PIL
- ✅ All other dependencies

## 🚀 Deployment Options

### Option 1: Streamlit Cloud (Recommended)

1. **Push to GitHub** (Already done ✅)
   - Repository: https://github.com/mohini-workday/Deepfake-Detection

2. **Deploy on Streamlit Cloud**:
   - Go to: https://share.streamlit.io/
   - Sign in with GitHub
   - Click "New app"
   - Select repository: `mohini-workday/Deepfake-Detection`
   - Main file path: `DeepFakeDetection_Dashboard.py`
   - Click "Deploy"

3. **Configuration**:
   - Streamlit Cloud will automatically detect `requirements.txt`
   - No additional configuration needed

### Option 2: Heroku

1. **Create `Procfile`**:
   ```
   web: streamlit run DeepFakeDetection_Dashboard.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Create `setup.sh`** (if needed):
   ```bash
   mkdir -p ~/.streamlit/
   echo "[server]" > ~/.streamlit/config.toml
   echo "headless = true" >> ~/.streamlit/config.toml
   ```

3. **Deploy**:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### Option 3: Docker

1. **Create `Dockerfile`**:
   ```dockerfile
   FROM python:3.10-slim
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   COPY . .
   
   EXPOSE 8501
   
   HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
   
   ENTRYPOINT ["streamlit", "run", "DeepFakeDetection_Dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Build and run**:
   ```bash
   docker build -t deepfake-dashboard .
   docker run -p 8501:8501 deepfake-dashboard
   ```

## ⚠️ Important Notes for Deployment

### Model Loading

The dashboard currently loads models with pretrained weights. For deployment:

1. **Option A**: Train models and save checkpoints, then load them
2. **Option B**: Use pretrained weights (current implementation)
3. **Option C**: Load models from a cloud storage (S3, GCS, etc.)

### Video Processing

- Videos are processed in memory (temporary files)
- Large videos may require more memory
- Consider adding file size limits in the uploader

### Performance Considerations

- Model loading is cached with `@st.cache_resource`
- First load may be slow (downloading pretrained weights)
- Consider pre-loading models or using smaller models for faster startup

## 📋 Deployment Checklist

- [x] All code pushed to GitHub
- [x] `requirements.txt` includes all dependencies
- [x] Dashboard file is complete and functional
- [ ] Models trained and saved (optional, for better accuracy)
- [ ] Test deployment locally first
- [ ] Configure environment variables if needed
- [ ] Set up monitoring/logging (optional)

## 🔧 Local Testing Before Deployment

```bash
# Activate virtual environment
source deepfake_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run DeepFakeDetection_Dashboard.py
```

## 📝 Current Status

✅ **Code Status**: All code is pushed to GitHub
✅ **Dependencies**: All in requirements.txt
✅ **File Structure**: Complete
⚠️ **Models**: Using pretrained weights (models will download on first use)
✅ **Ready for Deployment**: Yes, with notes above

## 🎯 Recommended Deployment Steps

1. **Test locally** - Ensure everything works
2. **Deploy to Streamlit Cloud** - Easiest option
3. **Monitor performance** - Check memory usage
4. **Optimize if needed** - Add model caching, file size limits

## 📞 Support

If you encounter issues during deployment:
1. Check Streamlit Cloud logs
2. Verify all dependencies in requirements.txt
3. Ensure Python version compatibility (3.8+)
4. Check model loading paths

