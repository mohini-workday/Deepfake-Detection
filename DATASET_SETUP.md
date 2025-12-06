# Dataset Setup Guide - Google Drive

## Dataset Structure

The dataset is organized in Google Drive with three folders:

1. **Celeb-Real** - Contains real/original videos
   - Label: `0`
   - Label Name: `"Celeb-Real"`

2. **Celeb-Fake** - Contains fake/manipulated videos
   - Label: `1`
   - Label Name: `"Fake"`

3. **Testing** - Contains test videos for evaluation
   - No label (for inference only)

## Google Drive Link

**Dataset Location**: https://drive.google.com/drive/folders/1nBKjUpi2wQyMfWDuNsreqY11DVZrbk7x

## Setup Instructions

### Option 1: Manual Download (Recommended for Large Datasets)

1. **Open the Google Drive link**:
   ```
   https://drive.google.com/drive/folders/1nBKjUpi2wQyMfWDuNsreqY11DVZrbk7x
   ```

2. **Download the three folders**:
   - Celeb-Real
   - Celeb-Fake
   - Testing

3. **Extract and place in project directory**:
   ```
   /path/to/project/data/
   ├── Celeb-Real/
   │   ├── video1.mp4
   │   ├── video2.mp4
   │   └── ...
   ├── Celeb-Fake/
   │   ├── video1.mp4
   │   ├── video2.mp4
   │   └── ...
   └── Testing/
       ├── test_video1.mp4
       ├── test_video2.mp4
       └── ...
   ```

4. **Run the notebook** - The code will automatically detect the folders

### Option 2: Automatic Download (May Not Work for Large Folders)

The notebook will attempt to download automatically using `gdown`, but this may fail for large folders. If automatic download fails, use Option 1.

## Label Assignment

Videos are automatically labeled based on their folder location:

- **Celeb-Real folder** → Label `0` ("Celeb-Real")
- **Celeb-Fake folder** → Label `1` ("Fake")
- **Testing folder** → No label (for inference)

## Supported Video Formats

The code supports the following video formats:
- `.mp4`
- `.avi`
- `.mov`
- `.mkv`
- `.flv`
- `.wmv`
- `.webm`

## Verification

After setup, run Cell 2 in the notebook. You should see:

```
✅ Dataset folders found locally!
✅ Training videos loaded: X
✅ Test videos loaded: Y
```

## Troubleshooting

### Issue: "Dataset folders not found locally"

**Solution**: 
1. Check that folders are in the `data/` directory
2. Verify folder names are exactly: `Celeb-Real`, `Celeb-Fake`, `Testing`
3. Ensure folders contain video files

### Issue: "No training videos found"

**Solution**:
1. Check video file formats (must be supported formats)
2. Verify videos are not corrupted
3. Check folder permissions

### Issue: Download fails

**Solution**: 
- Use manual download (Option 1)
- Large folders may require manual download
- Check internet connection

## Dataset Statistics

After loading, the notebook will display:
- Total number of training videos
- Label distribution (Celeb-Real vs Fake)
- Sample video paths
- Visualization of label distribution

## Next Steps

1. Download the dataset folders
2. Place them in the `data/` directory
3. Run Cell 2 in the notebook to load and verify the dataset
4. Proceed with model training


