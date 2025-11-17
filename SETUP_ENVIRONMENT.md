# Environment Setup Guide

## Quick Start: Virtual Environment Setup

### Option 1: Using venv (Recommended for Beginners)

```bash
# 1. Navigate to project folder
cd C:\Users\YourName\Downloads\dia

# 2. Create virtual environment
python -m venv venv

# 3. Activate it
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# 4. Install packages
pip install -r requirements.txt

# 5. Install dia package
cd dia
pip install -e .
cd ..
```

### Option 2: Using Conda

```bash
# 1. Navigate to project folder
cd C:\Users\YourName\Downloads\dia

# 2. Create conda environment
conda create -n dia_env python=3.10

# 3. Activate it
conda activate dia_env

# 4. Install packages
pip install -r requirements.txt

# 5. Install dia package
cd dia
pip install -e .
cd ..
```

## Requirements Files

- **requirements.txt** - For CPU version (works everywhere)
- **requirements_gpu.txt** - For GPU version (requires NVIDIA GPU with CUDA)

## Verifying Installation

```bash
# Check Python version
python --version

# Check if packages installed
pip list

# Check if GPU available (if using GPU version)
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
```

## Troubleshooting

**Problem: "venv: command not found"**
- Make sure Python is installed
- Use: `python -m venv venv`

**Problem: "conda: command not found"**
- Install Anaconda or Miniconda from https://www.anaconda.com/

**Problem: Packages not found after installation**
- Make sure virtual environment is activated
- Check you're using the correct Python interpreter

