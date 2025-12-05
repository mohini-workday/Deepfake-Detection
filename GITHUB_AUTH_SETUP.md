# GitHub Authentication Setup Guide

## Current Status ✅

- Git is configured with your credentials
- SSH keys exist on your system
- Remote repository changed to use SSH
- Ready to push!

## SSH Key Setup

Your SSH public key is:
```
<see output from terminal>
```

### If SSH key is NOT added to GitHub:

1. **Copy your SSH public key:**
   ```bash
   cat ~/.ssh/id_ed25519.pub | pbcopy
   ```
   (This copies it to your clipboard)

2. **Add to GitHub:**
   - Go to: https://github.com/settings/keys
   - Click "New SSH key"
   - Title: "Mac - DeepFake Detection"
   - Key: Paste the key from clipboard
   - Click "Add SSH key"

3. **Test connection:**
   ```bash
   ssh -T git@github.com
   ```
   You should see: "Hi mohini-workday! You've successfully authenticated..."

## Push to GitHub

Once SSH is set up, simply run:

```bash
cd "/Users/mohini.gangaram/Desktop/MLPostGrad/Sem3/Deep Learning/Final Project"
git push -u origin main
```

## Alternative: Personal Access Token (if SSH doesn't work)

If SSH authentication fails, you can use a Personal Access Token:

1. **Create Token:**
   - Go to: https://github.com/settings/tokens
   - Click "Generate new token (classic)"
   - Name: "DeepFake Detection Project"
   - Select scope: `repo` (full control)
   - Click "Generate token"
   - **Copy the token immediately** (you won't see it again!)

2. **Change remote back to HTTPS:**
   ```bash
   git remote set-url origin https://github.com/mohini-workday/Deepfake-Detection.git
   ```

3. **Push with token:**
   ```bash
   git push -u origin main
   ```
   - Username: `mohini-workday`
   - Password: `<paste your token>`

## Troubleshooting

### If SSH connection fails:
```bash
# Test SSH connection
ssh -T git@github.com

# If it says "Permission denied", add your SSH key to GitHub
# If it says "Hi mohini-workday!", you're good to go!
```

### If you get "remote: Support for password authentication was removed":
- You MUST use either SSH or Personal Access Token
- HTTPS with password no longer works

### Clear cached credentials (if needed):
```bash
git credential-osxkeychain erase
host=github.com
protocol=https
[Press Enter twice]
```

