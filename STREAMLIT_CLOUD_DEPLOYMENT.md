# Streamlit Cloud Deployment Guide

## 🔧 Fix: "App is not connected to GitHub repository"

This guide will help you properly connect your Deepfake Dashboard app to Streamlit Cloud.

## ✅ Pre-Deployment Checklist

### 1. Verify Repository Connection
Your repository is already connected to GitHub:
- **Repository**: `mohini-workday/Deepfake-Detection`
- **Remote URL**: `git@github.com:mohini-workday/Deepfake-Detection.git`

### 2. Required Files (All Present ✅)
- ✅ `DeepFakeDetection_Dashboard.py` - Main Streamlit app
- ✅ `requirements.txt` - All dependencies
- ✅ `.streamlit/config.toml` - Streamlit configuration (just created)
- ✅ `README.md` - Project documentation

### 3. Commit and Push Latest Changes

```bash
cd "/Users/mohini.gangaram/Desktop/MLPostGrad/Sem3/Deep Learning/Final Project"

# Add the new config file
git add .streamlit/config.toml

# Commit any changes
git add -A
git commit -m "Add Streamlit Cloud configuration"

# Push to GitHub
git push origin main
```

## 🚀 Step-by-Step Deployment on Streamlit Cloud

### Step 1: Access Streamlit Cloud
1. Go to **https://share.streamlit.io/**
2. Sign in with your **GitHub account** (use the same account: `mohini-workday`)

### Step 2: Authorize Streamlit Cloud
1. If prompted, authorize Streamlit Cloud to access your GitHub repositories
2. Grant permissions to:
   - Read repository contents
   - Access repository metadata
   - Deploy applications

### Step 3: Create New App
1. Click **"New app"** button (top right)
2. You'll see a form with the following fields:

### Step 4: Configure Your App
Fill in the deployment form:

**Repository:**
- Select: `mohini-workday/Deepfake-Detection`
- If you don't see it, click "Refresh" or check GitHub permissions

**Branch:**
- Select: `main` (or `master` if that's your default branch)

**Main file path:**
- Enter: `DeepFakeDetection_Dashboard.py`
- This is the path to your Streamlit app file

**App URL (optional):**
- Leave blank or customize: `deepfake-detection-dashboard`
- Final URL will be: `https://deepfake-detection-dashboard.streamlit.app`

**Python version:**
- Select: `3.10` or `3.11` (recommended)

### Step 5: Deploy
1. Click **"Deploy"** button
2. Streamlit Cloud will:
   - Clone your repository
   - Install dependencies from `requirements.txt`
   - Start your app
   - Provide you with a live URL

### Step 6: Monitor Deployment
- Watch the deployment logs in real-time
- Check for any errors
- If successful, you'll see: "Your app is live!"

## 🔍 Troubleshooting

### Issue: "App is not connected to GitHub repository"

**Solution 1: Reconnect Repository**
1. Go to your app settings in Streamlit Cloud
2. Click "Settings" → "Repository"
3. Click "Disconnect" and then reconnect
4. Select the correct repository: `mohini-workday/Deepfake-Detection`

**Solution 2: Check GitHub Permissions**
1. Go to GitHub Settings: https://github.com/settings/applications
2. Find "Streamlit Cloud" in authorized applications
3. Ensure it has access to your repositories
4. If not, revoke and re-authorize

**Solution 3: Verify Repository Access**
```bash
# Test if you can access the repository
cd "/Users/mohini.gangaram/Desktop/MLPostGrad/Sem3/Deep Learning/Final Project"
git remote -v
# Should show: origin  git@github.com:mohini-workday/Deepfake-Detection.git

# Test GitHub connection
ssh -T git@github.com
# Should show: Hi mohini-workday! You've successfully authenticated...
```

**Solution 4: Check Repository Visibility**
- Ensure the repository is **public** OR
- If private, ensure Streamlit Cloud has access:
  1. Go to repository settings on GitHub
  2. Settings → Collaborators → Add Streamlit Cloud
  3. Or make repository public temporarily for deployment

### Issue: "Module not found" or Import Errors

**Solution:**
- Verify all dependencies are in `requirements.txt`
- Check that package names are correct (case-sensitive)
- Ensure version numbers are compatible

### Issue: "File not found" Errors

**Solution:**
- Ensure all required files are committed to GitHub
- Check file paths in your code (use relative paths)
- Verify `.gitignore` isn't excluding necessary files

## 📋 Post-Deployment Checklist

- [ ] App is accessible via the provided URL
- [ ] All pages/sections load correctly
- [ ] Video upload functionality works
- [ ] Model loading works (may take time on first load)
- [ ] All visualizations display correctly
- [ ] No errors in the app logs

## 🔄 Updating Your Deployed App

After making changes to your code:

```bash
cd "/Users/mohini.gangaram/Desktop/MLPostGrad/Sem3/Deep Learning/Final Project"

# Commit changes
git add -A
git commit -m "Update dashboard features"

# Push to GitHub
git push origin main
```

Streamlit Cloud will **automatically redeploy** your app when you push to the main branch!

## 📞 Additional Resources

- **Streamlit Cloud Docs**: https://docs.streamlit.io/streamlit-community-cloud
- **Streamlit Cloud Status**: https://status.streamlit.io/
- **Community Forum**: https://discuss.streamlit.io/

## ✅ Quick Reference

**Repository URL**: https://github.com/mohini-workday/Deepfake-Detection
**Main App File**: `DeepFakeDetection_Dashboard.py`
**Requirements**: `requirements.txt`
**Config**: `.streamlit/config.toml`

---

**Need Help?** Check the deployment logs in Streamlit Cloud dashboard for specific error messages.

