# Quick Start Guide - Command Line Reference

## Complete Setup (First Time)

```bash
# 1. Navigate to project folder
cd C:\Users\YourName\Downloads\dia

# 2. Create virtual environment
python -m venv venv

# 3. Activate environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# 4. Set Hugging Face token
python setup_hf_token.py
# OR manually:
set HF_TOKEN=your_token_here  # Windows
export HF_TOKEN=your_token_here  # Mac/Linux

# 5. Install packages
pip install -r requirements.txt

# 6. Install dia package
cd dia
pip install -e .
cd ..

# 7. Prepare dataset (downloads from Hugging Face)
python prepare_training_data.py

# 8. Run test (downloads model from Hugging Face)
python test_pretrained_model.py
```

## Daily Use (After Setup)

```bash
# 1. Navigate to project
cd C:\Users\YourName\Downloads\dia

# 2. Activate environment
venv\Scripts\activate

# 3. Set token (if not permanent)
set HF_TOKEN=your_token_here

# 4. Run test
python test_pretrained_model.py
```

## GPU Version

```bash
# 1. Navigate to GPU folder
cd C:\Users\YourName\Downloads\dia\GPU_version

# 2. Activate environment
..\venv\Scripts\activate

# 3. Set token
set HF_TOKEN=your_token_here

# 4. Run GPU test
python test_pretrained_model_gpu.py
```

## Key Commands

**Check if token is set:**
```bash
python -c "import os; print('Token:', 'HF_TOKEN' in os.environ)"
```

**Check if GPU is available:**
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

**List installed packages:**
```bash
pip list
```

**Deactivate environment:**
```bash
deactivate
```

## Hugging Face Resources

- **Model:** https://huggingface.co/nari-labs/Dia-1.6B-0626
- **Dataset:** https://huggingface.co/datasets/deepvk/NonverbalTTS
- **Create Token:** https://huggingface.co/settings/tokens

