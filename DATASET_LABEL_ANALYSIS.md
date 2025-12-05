# Dataset Label Analysis Guide

## Question: Does the dataset contain both Original and Manipulated videos?

Based on the dataset name **"Hemgg/deep-fake-detection-dfd-entire-original-dataset"**, there's a possibility that this dataset contains **only original videos**, not manipulated ones.

## How to Check

I've added code cells in the notebook that will:

1. **Load the first 200 samples** from the dataset
2. **Extract labels from file paths** using the `infer_label_from_path()` function
3. **Analyze the label distribution** and show:
   - Count of Original vs Manipulated videos
   - Percentage distribution
   - Sample file paths to verify

## Expected Scenarios

### Scenario 1: Dataset contains ONLY Original videos
- **Indicator**: All 200 samples will be labeled as "Original" (label = 0)
- **File paths**: Will likely contain words like "original", "pristine", "real"
- **Action needed**: 
  - You'll need to combine with another dataset that has manipulated/fake videos
  - Or use a different dataset that has both classes

### Scenario 2: Dataset contains BOTH classes
- **Indicator**: You'll see both "Original" and "Manipulated" labels
- **File paths**: Will contain both "original_sequences" and "manipulated_sequences" or similar
- **Action**: You can proceed with training as-is

## Running the Analysis

1. Run **Cell 2** (Data Loading and Label Analysis) - This will:
   - Load 200 samples
   - Analyze labels from file paths
   - Display distribution charts
   - Show warnings if only one class is found

2. Run **Cell 3** (Deep Dataset Structure Analysis) - This will:
   - Check the internal structure of samples
   - Look for any hidden label fields
   - Display all available keys in the dataset

## Alternative Datasets

If this dataset only contains original videos, consider these alternatives:

1. **DFDC (Deepfake Detection Challenge) Dataset**
   - Contains both original and manipulated videos
   - Available on Kaggle or AWS

2. **FaceForensics++**
   - Contains original and manipulated videos
   - Multiple manipulation methods

3. **Celeb-DF**
   - High-quality deepfake dataset
   - Both original and fake videos

## Label Inference Logic

The code uses file path patterns to infer labels:
- **Original (0)**: Paths containing "original_sequences", "pristine", "original"
- **Manipulated (1)**: Paths containing "manipulated_sequences", "dfdc", "fake", "deepfake"

If paths don't match these patterns, they default to "Original" (0).

## Next Steps

After running the analysis cells:

1. **Check the output** - Look for the label distribution
2. **Review sample paths** - Verify the path structure
3. **If only one class**: 
   - Find a complementary dataset with manipulated videos
   - Or use a different dataset entirely
4. **If both classes**: Proceed with the full pipeline

## Important Note

The dataset name suggests it might be the "original" subset. The full DFD (DeepFake Detection) dataset typically has:
- Original videos dataset
- Manipulated videos dataset (separate)

You may need to load both datasets and combine them.

