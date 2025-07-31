# Data Management Guide

This document provides guidelines for managing large files and experimental data in this project.

## 🚫 What NOT to Upload to GitHub

### Model Weights and Checkpoints
- `*.pth`, `*.pt`, `*.bin`, `*.safetensors`
- `checkpoints/`, `models/`
- Pre-trained model files (usually > 100MB)

### Experimental Results
- `experiments/*/` - All experiment directories
- `outputs/`, `results/`, `runs/`
- Large log files (`*.log`, `*.out`, `*.err`)
- WandB files (`wandb/`)

### Large Datasets
- `data/`, `datasets/`, `images/`
- Raw image files (`.jpg`, `.png`, etc.)
- Large JSON/CSV files containing embeddings or features
- Compressed archives (`.zip`, `.tar.gz`, etc.)

## ✅ What TO Upload

### Code and Configuration
- All Python source code (`.py` files)
- Configuration files (`configs/*.yaml`, `configs/*.json`)
- Documentation (`.md` files)
- Requirements and setup files

### Small Example Files
- Sample data for testing (< 1MB)
- Example configuration files
- Small hard negative samples for documentation

## 📁 Recommended Project Structure

```
MyComposedRetrieval/
├── src/                    # ✅ Upload: Source code
├── configs/                # ✅ Upload: Configuration files
├── scripts/                # ✅ Upload: Training/evaluation scripts
├── requirements.txt        # ✅ Upload: Dependencies
├── README.md              # ✅ Upload: Documentation
├── .gitignore             # ✅ Upload: Git configuration
│
├── data/                  # ❌ Local only: Large datasets
├── experiments/           # ❌ Local only: Experiment results
├── checkpoints/           # ❌ Local only: Model checkpoints
├── models/                # ❌ Local only: Pre-trained models
├── outputs/               # ❌ Local only: Generated outputs
└── wandb/                 # ❌ Local only: WandB logs
```

## 💾 Alternative Storage Solutions

### For Model Weights
1. **Hugging Face Hub**: Upload trained models to HF Hub
2. **Google Drive/OneDrive**: Share via cloud storage
3. **University Storage**: Use institutional storage systems

### For Experimental Results
1. **WandB**: Track experiments online
2. **TensorBoard**: Log metrics and visualizations
3. **Local Archives**: Keep results in local compressed files

### For Large Datasets
1. **Download Scripts**: Provide scripts to download public datasets
2. **Dataset Documentation**: Document data sources and preprocessing steps
3. **Sample Data**: Include small subsets for testing

## 🔧 Best Practices

### Before Committing
```bash
# Check what you're about to commit
git status
git diff --cached

# Make sure no large files are included
find . -size +50M -type f
```

### Cleaning Up Accidentally Committed Files
```bash
# Remove large files from Git history
git filter-branch --tree-filter 'rm -rf path/to/large/file' HEAD
# or use BFG Repo-Cleaner for better performance
```

### Using Git LFS (for selective large files)
```bash
# Install Git LFS
git lfs install

# Track specific file types
git lfs track \"*.pth\"
git lfs track \"*.bin\"

# Add .gitattributes to repo
git add .gitattributes
```

## 📊 Repository Size Guidelines

- **Total repository size**: < 100MB (recommended)
- **Individual file size**: < 10MB (GitHub soft limit: 100MB)
- **Keep it lean**: Only commit essential code and documentation

## 🔍 Monitoring Repository Size

```bash
# Check repository size
du -sh .git

# Find largest files
git ls-files | xargs ls -l | sort -nrk5 | head -10

# Check which files are tracked
git ls-files --cached | wc -l
```

Remember: **Code is permanent, experiments are temporary** - Keep your repository focused on reproducible code rather than experimental artifacts.