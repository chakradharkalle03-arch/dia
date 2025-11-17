# Dia Model - User Manual
## Complete Guide for Non-Technical Users

**Version:** 1.0  
**Date:** November 2025  
**Model:** Dia-1.6B Fine-tuned on NonverbalTTS Dataset

---

## Table of Contents

1. [Introduction](#introduction)
2. [What You Need](#what-you-need)
3. [Installation Guide](#installation-guide)
4. [Using the Pretrained Model](#using-the-pretrained-model)
5. [Generating Audio](#generating-audio)
6. [Troubleshooting](#troubleshooting)
7. [FAQ](#faq)

---

## Introduction

This manual will help you use the Dia text-to-speech model that has been trained on the NonverbalTTS dataset. This model can convert text into natural-sounding speech, including nonverbal sounds like breaths, laughs, coughs, and more.

**What this model can do:**
- Convert text to speech
- Generate natural-sounding voices
- Include nonverbal sounds (breaths, laughs, etc.) in speech
- Support multiple speakers

**Who is this for:**
- Anyone who wants to generate speech from text
- Users with no coding experience
- People who want to use AI voice generation

---

## What You Need

### System Requirements

**Minimum Requirements:**
- **Operating System:** Windows 10 or later, macOS 10.14+, or Linux
- **RAM:** 8 GB minimum (16 GB recommended)
- **Storage:** 10 GB free space
- **Internet:** Required for initial setup (to download the model)

**For GPU Version (Faster):**
- NVIDIA GPU with 4GB+ VRAM
- CUDA-compatible graphics card

**For CPU Version (Slower but works everywhere):**
- Any modern computer
- More patience (will be slower)

### Software Requirements

1. **Python 3.8 or higher**
   - Download from: https://www.python.org/downloads/
   - Make sure to check "Add Python to PATH" during installation

2. **Git** (optional, for downloading files)
   - Download from: https://git-scm.com/downloads

3. **Text Editor** (for editing configuration files)
   - Windows: Notepad (built-in)
   - Mac: TextEdit (built-in)
   - Or download Notepad++: https://notepad-plus-plus.org/

4. **Hugging Face Account** (Required)
   - Create free account at: https://huggingface.co/join
   - You'll need this to download the model and dataset

---

## Installation Guide

### Step 1: Install Python

1. Go to https://www.python.org/downloads/
2. Click "Download Python" (latest version)
3. Run the installer
4. **IMPORTANT:** Check the box "Add Python to PATH"
5. Click "Install Now"
6. Wait for installation to complete
7. Click "Close"

**Verify Installation:**
- Open Command Prompt (Windows) or Terminal (Mac/Linux)
- Type: `python --version`
- You should see something like: `Python 3.12.x`

### Step 2: Set Up Hugging Face API Token

**Why you need this:**
- The model and dataset are hosted on Hugging Face
- You need an API token to download them
- This is free and takes 2 minutes

**Step-by-Step:**

1. **Create Hugging Face Account**
   - Go to: https://huggingface.co/join
   - Sign up with email or GitHub account
   - Verify your email address

2. **Create API Token**
   - Log in to Hugging Face
   - Go to: https://huggingface.co/settings/tokens
   - Click "New token"
   - Name it: "Dia Model Access"
   - Select "Read" permission (this is enough)
   - Click "Generate token"
   - **IMPORTANT:** Copy the token immediately (you won't see it again!)
   - It looks like: `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

3. **Set Up Token on Your Computer**

   **Option A: Using Environment Variable (Recommended)**
   
   **Windows:**
   ```
   # In Command Prompt (temporary - for current session)
   set HF_TOKEN=your_token_here
   
   # Or set permanently:
   # 1. Right-click "This PC" → Properties
   # 2. Advanced system settings → Environment Variables
   # 3. New → Variable name: HF_TOKEN, Value: your_token_here
   ```
   
   **Mac/Linux:**
   ```
   # In Terminal (temporary - for current session)
   export HF_TOKEN=your_token_here
   
   # Or add to ~/.bashrc or ~/.zshrc (permanent):
   echo 'export HF_TOKEN=your_token_here' >> ~/.bashrc
   source ~/.bashrc
   ```
   
   **Option B: Using Python Script (Easier)**
   
   Create a file named `setup_hf_token.py`:
   ```python
   import os
   
   print("=" * 70)
   print("Hugging Face Token Setup")
   print("=" * 70)
   
   token = input("\nEnter your Hugging Face token: ").strip()
   
   if not token.startswith("hf_"):
       print("Warning: Token should start with 'hf_'")
   
   # Set environment variable for current session
   os.environ["HF_TOKEN"] = token
   
   print("\n[OK] Token set for current session")
   print("\nTo make it permanent:")
   print("Windows: setx HF_TOKEN \"" + token + "\"")
   print("Mac/Linux: Add 'export HF_TOKEN=\"" + token + "\"' to ~/.bashrc")
   ```
   
   Run it:
   ```
   python setup_hf_token.py
   ```

4. **Verify Token Works**
   ```
   python -c "import os; print('Token set:', 'HF_TOKEN' in os.environ)"
   ```

**Important Notes:**
- Keep your token secret (like a password)
- Don't share it publicly
- If you lose it, create a new one from Hugging Face settings
- The token is free and unlimited for personal use

### Step 3: Download the Project Files

**Option A: Download as ZIP**
1. Download all project files as a ZIP folder
2. Extract the ZIP file to a location like `C:\Users\YourName\Downloads\dia`
3. Remember this location

**Option B: Using Git (if installed)**
1. Open Command Prompt/Terminal
2. Navigate to where you want the files:
   ```
   cd C:\Users\YourName\Downloads
   ```
3. Run:
   ```
   git clone [repository-url]
   ```

### Step 4: Set Up Virtual Environment (Recommended)

**Why use a virtual environment?**
- Keeps your project packages separate from other projects
- Prevents conflicts between different package versions
- Makes it easier to manage dependencies

**Option A: Using venv (Built into Python)**

1. Open Command Prompt (Windows) or Terminal (Mac/Linux)

2. Navigate to the project folder:
   ```
   cd C:\Users\YourName\Downloads\dia
   ```

3. Create virtual environment:
   ```
   python -m venv venv
   ```

4. Activate virtual environment:
   - **Windows:**
     ```
     venv\Scripts\activate
     ```
   - **Mac/Linux:**
     ```
     source venv/bin/activate
     ```

5. You should see `(venv)` at the start of your command line

6. To deactivate later, just type: `deactivate`

**Option B: Using Conda (Anaconda/Miniconda)**

1. Open Anaconda Prompt (Windows) or Terminal (Mac/Linux)

2. Navigate to the project folder:
   ```
   cd C:\Users\YourName\Downloads\dia
   ```

3. Create conda environment:
   ```
   conda create -n dia_env python=3.10
   ```

4. Activate conda environment:
   ```
   conda activate dia_env
   ```

5. You should see `(dia_env)` at the start of your command line

6. To deactivate later, type: `conda deactivate`

**Note:** After creating the environment, continue with Step 5 below.

### Step 5: Install Required Packages

**Method 1: Using requirements.txt (Easiest)**

1. Make sure your virtual environment is activated (you should see `(venv)` or `(dia_env)`)

2. Navigate to the project folder:
   ```
   cd C:\Users\YourName\Downloads\dia
   ```

3. Install all packages at once:
   ```
   pip install -r requirements.txt
   ```

4. Wait for all packages to install (this may take 10-20 minutes)

**Method 2: Manual Installation**

If requirements.txt doesn't work, install packages manually:

1. Make sure your virtual environment is activated

2. Install packages one by one:
   ```
   pip install torch torchvision torchaudio
   pip install transformers datasets accelerate
   pip install soundfile librosa
   pip install tqdm numpy
   ```

**For GPU Users:**

If you have an NVIDIA GPU and want faster processing:

1. Use the GPU requirements file:
   ```
   pip install -r requirements_gpu.txt
   ```

2. Or install PyTorch with CUDA manually:
   ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

**Using Conda for GPU:**

If using conda, you can install PyTorch with CUDA:
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Step 6: Install the Dia Model Package

1. Make sure your virtual environment is activated

2. Still in the project folder, navigate to the dia subfolder:
   ```
   cd dia
   ```

3. Install the package:
   ```
   pip install -e .
   ```

4. Go back to the main folder:
   ```
   cd ..
   ```

### Step 7: Download Model from Hugging Face

**What is this?**
- The Dia model is hosted on Hugging Face
- You need to download it (automatic on first use)
- Size: ~3-4 GB, takes 10-30 minutes depending on internet speed

**Method 1: Automatic Download (Recommended)**

The model will download automatically when you first run the test script. No action needed!

**Method 2: Manual Download (If automatic fails)**

1. Make sure your HF_TOKEN is set (see Step 2)

2. Run this Python command:
   ```
   python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='nari-labs/Dia-1.6B-0626', token=os.environ.get('HF_TOKEN'))"
   ```

3. Or install huggingface_hub and download:
   ```
   pip install huggingface_hub
   python
   ```
   Then in Python:
   ```python
   import os
   from huggingface_hub import snapshot_download
   snapshot_download(repo_id='nari-labs/Dia-1.6B-0626', token=os.environ.get('HF_TOKEN'))
   ```

**Model Information:**
- **Model Name:** nari-labs/Dia-1.6B-0626
- **Location:** https://huggingface.co/nari-labs/Dia-1.6B-0626
- **Size:** ~3-4 GB
- **Download Location:** Usually `C:\Users\YourName\.cache\huggingface\hub\` (Windows)

### Step 8: Download Dataset from Hugging Face

**What is this?**
- The NonverbalTTS dataset is used for training
- You need it to prepare training data
- Size: ~500 MB, takes 5-10 minutes

**Method 1: Automatic Download (Recommended)**

The dataset will download automatically when you run the preparation script. No action needed!

**Method 2: Manual Download (If automatic fails)**

1. Make sure your HF_TOKEN is set

2. Run the dataset preparation script:
   ```
   python prepare_training_data.py
   ```

3. The script will:
   - Download dataset from: https://huggingface.co/datasets/deepvk/NonverbalTTS
   - Process and convert emojis to text tags
   - Save to `processed_data/processed_dataset.json`

**Dataset Information:**
- **Dataset Name:** deepvk/NonverbalTTS
- **Location:** https://huggingface.co/datasets/deepvk/NonverbalTTS
- **Size:** ~500 MB
- **Processed Output:** `processed_data/processed_dataset.json`

**What the script does:**
- Downloads the dataset
- Converts emoji tokens (🌬️, 🤣, etc.) to text tags ((breaths), (laughs), etc.)
- Formats text for Dia model
- Saves processed data for training

**Expected Output:**
```
======================================================================
Preparing NonverbalTTS Dataset for Training
======================================================================

1. Loading dataset...
[OK] Loaded 3641 samples

2. Processing dataset...
Processing: 100%|████████████| 3641/3641 [05:23<00:00]

[OK] Processed 3641 samples
  Skipped 0 samples

[OK] Saved processed dataset to: processed_data/processed_dataset.json

3. Dataset Statistics:
   Total samples: 3641
   Emotions:
     - neutral: 1200
     - happy: 800
     ...

======================================================================
Dataset Preparation Complete!
======================================================================
```

---

## Using the Pretrained Model

### Understanding the Folder Structure

After installation, you should have these folders:

```
dia/
├── GPU_version/              (For GPU users - faster)
│   ├── checkpoints/          (Trained model files)
│   ├── generated_audio/      (Output audio files)
│   └── test_pretrained_model_gpu.py
│
├── processed_data/           (Prepared dataset)
│   └── processed_dataset.json
│
└── dia/                      (Model code)
```

### Complete Command-Line Guide

**Important:** Always activate your virtual environment first!

#### Step-by-Step: Running CPU Version

1. **Open Command Prompt (Windows) or Terminal (Mac/Linux)**

2. **Navigate to project folder:**
   ```
   cd C:\Users\YourName\Downloads\dia
   ```
   (Replace with your actual path)

3. **Activate virtual environment:**
   ```
   venv\Scripts\activate
   ```
   You should see `(venv)` at the start of your command line

4. **Verify HF_TOKEN is set (if not already set):**
   ```
   python -c "import os; print('Token:', 'HF_TOKEN' in os.environ)"
   ```
   If it says `False`, set it:
   ```
   set HF_TOKEN=your_token_here
   ```

5. **Run the test script:**
   ```
   python test_pretrained_model.py
   ```

6. **What happens:**
   - First time: Downloads model (~3-4 GB, 10-30 minutes)
   - Model loads into memory (2-5 minutes)
   - Processes text
   - Generates audio
   - Saves to `generated_audio/test_output_[timestamp].wav`

7. **Expected output:**
   ```
   ======================================================================
   Dia Model - Pretrained Model Testing
   Fine-tuned on NonverbalTTS Dataset (50 epochs)
   ======================================================================
   ======================================================================
   Loading Pretrained Model
   ======================================================================
   
   Device: cpu
   
   Loading base model: nari-labs/Dia-1.6B-0626
   [OK] Base model loaded
   
   ======================================================================
   Generating Audio
   ======================================================================
   
   Generating audio...
   (This may take a moment...)
   generate: starting generation loop
   ...
   [SUCCESS] Audio generated and saved to: generated_audio\test_output_20251115_200046.wav
   ```

#### Step-by-Step: Running GPU Version

1. **Open Command Prompt (Windows) or Terminal (Mac/Linux)**

2. **Navigate to GPU_version folder:**
   ```
   cd C:\Users\YourName\Downloads\dia\GPU_version
   ```

3. **Activate virtual environment:**
   ```
   ..\venv\Scripts\activate
   ```
   Or if you created a separate environment:
   ```
   conda activate dia_env
   ```

4. **Verify HF_TOKEN is set:**
   ```
   python -c "import os; print('Token:', 'HF_TOKEN' in os.environ)"
   ```

5. **Run the GPU test script:**
   ```
   python test_pretrained_model_gpu.py
   ```

6. **What happens:**
   - Downloads model if first time (faster with GPU)
   - Model loads into GPU memory (30-60 seconds)
   - Processes text (much faster than CPU)
   - Generates audio
   - Saves to `generated_audio/test_output_gpu_[timestamp].wav`

### Command-Line Reference

**Full command sequence for first-time setup:**

```bash
# 1. Navigate to project
cd C:\Users\YourName\Downloads\dia

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Set Hugging Face token
set HF_TOKEN=your_token_here

# 4. Install packages
pip install -r requirements.txt

# 5. Install dia package
cd dia
pip install -e .
cd ..

# 6. Prepare dataset (downloads automatically)
python prepare_training_data.py

# 7. Run test (downloads model automatically)
python test_pretrained_model.py
```

**For subsequent runs (after setup):**

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

**For GPU version:**

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

---

## Generating Audio

### Method 1: Using the Default Test Script

The test scripts come with a default example. Simply run:

**For CPU:**
```
python test_pretrained_model.py
```

**For GPU:**
```
cd GPU_version
python test_pretrained_model_gpu.py
```

The script will:
1. Load the pretrained model
2. Process the example text
3. Generate audio
4. Save it to `generated_audio/` folder

**Output Location:**
- CPU version: `generated_audio/test_output_[timestamp].wav`
- GPU version: `GPU_version/generated_audio/test_output_gpu_[timestamp].wav`

### Method 2: Customizing the Text

To generate audio with your own text:

1. Open the test script file:
   - CPU: `test_pretrained_model.py`
   - GPU: `GPU_version/test_pretrained_model_gpu.py`

2. Find this line (around line 184):
   ```python
   test_text = "I tried to explain it to him..."
   ```

3. Replace the text with your own:
   ```python
   test_text = "Your text here. You can include emojis like 🌬️ for breaths or 🤣 for laughs."
   ```

4. Save the file

5. Run the script again:
   ```
   python test_pretrained_model.py
   ```

### Understanding Nonverbal Tokens

The model supports special tokens for nonverbal sounds:

| Emoji | Meaning | Text Tag |
|-------|---------|----------|
| 🌬️ | Breaths | (breaths) |
| 🤣 | Laughs | (laughs) |
| 😷 | Coughs | (coughs) |
| 👃 | Sniffs | (sniffs) |
| 🤧 | Sneezes | (sneezes) |
| 😤 | Grunts | (grunts) |
| 😖 | Groans | (groans) |

**How to use:**
- Include emojis in your text: `"Hello 🌬️ how are you?"`
- Or use text tags: `"Hello (breaths) how are you?"`
- The model will automatically convert emojis to tags

### Example Texts

**Example 1: Simple text**
```
test_text = "Hello, welcome to our presentation today."
```

**Example 2: With breaths**
```
test_text = "I mean, yeah, it makes me a little bit scared 🌬️ but it also makes me relaxed."
```

**Example 3: With laughs**
```
test_text = "That joke was hilarious 🤣 I couldn't stop laughing!"
```

**Example 4: Multiple nonverbal sounds**
```
test_text = "After running up the stairs 🌬️ I was out of breath 🤣 but happy!"
```

---

## Virtual Environment Management

### Creating a New Virtual Environment

**Using venv:**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

**Using conda:**
```bash
conda create -n dia_env python=3.10
conda activate dia_env
```

### Installing Packages in Virtual Environment

**Always activate your environment first!**

```bash
# Activate environment
venv\Scripts\activate  # or: conda activate dia_env

# Install from requirements.txt
pip install -r requirements.txt

# Or install GPU version
pip install -r requirements_gpu.txt
```

### Deactivating Virtual Environment

```bash
deactivate  # For venv
conda deactivate  # For conda
```

### Common Virtual Environment Issues

**Problem: "pip is not recognized"**
- Make sure virtual environment is activated
- Check that Python is installed correctly

**Problem: Packages installed but not found**
- Verify environment is activated
- Check you're using the correct Python interpreter

**Problem: "Command not found" (Mac/Linux)**
- Use `source venv/bin/activate` instead of just `venv/bin/activate`

## Troubleshooting

### Problem: "Python is not recognized"

**Solution:**
1. Python is not installed or not in PATH
2. Reinstall Python and make sure to check "Add Python to PATH"
3. Restart Command Prompt/Terminal after installation

### Problem: "Module not found" or "No module named..."

**Solution:**
1. Make sure virtual environment is activated
2. Missing required packages - install from requirements.txt:
   ```
   pip install -r requirements.txt
   ```
3. Or install specific package:
   ```
   pip install [module-name]
   ```
4. If using conda:
   ```
   conda install [package-name]
   ```

### Problem: "CUDA is not available" (GPU version)

**Solutions:**
- **Option 1:** Use CPU version instead (works on any computer)
- **Option 2:** Install PyTorch with CUDA support:
  ```
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```
- **Option 3:** Check if you have an NVIDIA GPU:
  - Windows: Open Device Manager → Display adapters
  - Look for "NVIDIA" in the name

### Problem: "Out of memory" error

**Solutions:**
1. Close other applications
2. Reduce batch size in training scripts
3. Use CPU version (slower but uses less memory)
4. Restart your computer

### Problem: Audio file not generated

**Check:**
1. Did the script complete without errors?
2. Check the `generated_audio/` folder
3. Look for files with `.wav` extension
4. Check the script output for error messages

### Problem: Model takes too long to load

**This is normal:**
- First load: 2-5 minutes (downloading model)
- Subsequent loads: 30-60 seconds
- Be patient, especially on CPU

### Problem: "FileNotFoundError" or "Path not found"

**Solution:**
1. Make sure you're in the correct folder
2. Check the file path in the error message
3. Navigate to the correct folder:
   ```
   cd C:\Users\YourName\Downloads\dia
   ```
   (Use your actual path)

---

## FAQ

### Q: Do I need to use a virtual environment?

**A:** Highly recommended! Virtual environments:
- Keep your project isolated from other Python projects
- Prevent package conflicts
- Make it easier to manage dependencies
- Can be easily deleted and recreated if something goes wrong

### Q: Should I use venv or conda?

**A:** 
- **venv**: Built into Python, simpler, good for most users
- **conda**: Better for scientific computing, handles complex dependencies better
- Either works fine for this project!

### Q: Do I need to train the model myself?

**A:** No! The model is already trained. You just need to:
1. Set up virtual environment (recommended)
2. Install the software
3. Run the test script
4. Generate audio

### Q: How long does it take to generate audio?

**A:** 
- CPU: 30 seconds to 2 minutes per audio file
- GPU: 5-15 seconds per audio file
- First time loading model: 2-5 minutes (one-time)

### Q: Can I use this commercially?

**A:** Check the license of:
- The Dia model (Nari-Labs)
- The NonverbalTTS dataset
- Contact the original authors for commercial use

### Q: What audio formats are supported?

**A:** 
- Input: Text only
- Output: WAV files (44.1 kHz, mono)

### Q: Can I change the voice or speaker?

**A:** Yes! In the test script, find:
```python
speaker_id="S1"
```
Change to:
```python
speaker_id="S2"  # or other speaker IDs
```

### Q: How do I make the audio longer or shorter?

**A:** Adjust the `max_tokens` parameter in the test script:
```python
max_tokens=2000  # Increase for longer audio, decrease for shorter
```

### Q: Can I batch process multiple texts?

**A:** Yes, but you'll need to modify the script:
1. Create a list of texts
2. Loop through them
3. Generate audio for each

### Q: The audio quality is not good. How to improve?

**A:** 
- Use GPU version for better performance
- Increase `max_tokens` for longer context
- Adjust `temperature` (lower = more consistent, higher = more varied)
- Adjust `cfg_scale` (higher = more controlled)

### Q: Where are my generated audio files?

**A:**
- CPU version: `generated_audio/` folder in main directory
- GPU version: `GPU_version/generated_audio/` folder
- Files are named: `test_output_[timestamp].wav`

### Q: How do I get the Hugging Face token?

**A:**
1. Create account at https://huggingface.co/join
2. Go to https://huggingface.co/settings/tokens
3. Click "New token"
4. Name it and select "Read" permission
5. Copy the token (starts with `hf_`)

### Q: Where is the model downloaded to?

**A:**
- **Windows:** `C:\Users\YourName\.cache\huggingface\hub\models--nari-labs--Dia-1.6B-0626`
- **Mac/Linux:** `~/.cache/huggingface/hub/models--nari-labs--Dia-1.6B-0626`
- The model is cached, so it only downloads once

### Q: Do I need internet every time?

**A:**
- **First time:** Yes (to download model and dataset)
- **After that:** No, everything is cached locally
- You can work offline after initial setup

### Q: The model download is slow. Can I speed it up?

**A:**
- Use a faster internet connection
- Download during off-peak hours
- The model is ~3-4 GB, so it takes time
- Once downloaded, it's cached and won't download again

### Q: How do I know if the model downloaded successfully?

**A:**
- Check the cache folder (see above)
- Look for files ending in `.safetensors` or `.bin`
- The script will show "[OK] Model loaded" if successful
- If download fails, you'll see an error message

### Q: Can I use this offline?

**A:** 
- After initial setup: Yes (model is downloaded locally)
- First time: No (needs internet to download model and packages)

### Q: How much disk space do I need?

**A:**
- Model files: ~3-4 GB
- Dataset: ~500 MB
- Generated audio: ~1 MB per minute
- Total: ~5-6 GB recommended

---

## Quick Reference

### Essential Commands

**Navigate to project folder:**
```
cd C:\Users\YourName\Downloads\dia
```

**Activate virtual environment:**
```
venv\Scripts\activate  # Windows venv
source venv/bin/activate  # Mac/Linux venv
conda activate dia_env  # Conda
```

**Install all packages:**
```
pip install -r requirements.txt  # CPU version
pip install -r requirements_gpu.txt  # GPU version
```

**Run CPU version:**
```
python test_pretrained_model.py
```

**Run GPU version:**
```
cd GPU_version
python test_pretrained_model_gpu.py
```

**Check Python version:**
```
python --version
```

**Check if GPU is available:**
```
python -c "import torch; print(torch.cuda.is_available())"
```

**Deactivate virtual environment:**
```
deactivate  # venv
conda deactivate  # conda
```

### File Locations

| Item | Location |
|------|----------|
| CPU Test Script | `test_pretrained_model.py` |
| GPU Test Script | `GPU_version/test_pretrained_model_gpu.py` |
| Generated Audio (CPU) | `generated_audio/` |
| Generated Audio (GPU) | `GPU_version/generated_audio/` |
| Model Checkpoints | `checkpoints/dia-nonverbal-lora/` or `GPU_version/checkpoints/dia-nonverbal-lora-gpu/` |
| Dataset | `processed_data/processed_dataset.json` |
| Model Cache (Windows) | `C:\Users\YourName\.cache\huggingface\hub\` |
| Model Cache (Mac/Linux) | `~/.cache/huggingface/hub/` |
| Requirements File (CPU) | `requirements.txt` |
| Requirements File (GPU) | `requirements_gpu.txt` |

---

## Support and Additional Resources

### Getting Help

1. **Check this manual first** - Most common issues are covered
2. **Check error messages** - They often tell you what's wrong
3. **Check the script output** - Look for warnings or errors

### Useful Links

- Python Official: https://www.python.org/
- PyTorch Documentation: https://pytorch.org/docs/
- Dia Model: https://huggingface.co/nari-labs/Dia-1.6B-0626
- NonverbalTTS Dataset: https://huggingface.co/datasets/deepvk/NonverbalTTS

### Common Error Messages and Solutions

| Error | Solution |
|-------|----------|
| `python: command not found` | Install Python and add to PATH |
| `ModuleNotFoundError` | Run `pip install [module-name]` |
| `CUDA out of memory` | Use CPU version or close other apps |
| `FileNotFoundError` | Check you're in the correct folder |
| `Permission denied` | Run as administrator (Windows) or use `sudo` (Mac/Linux) |

---

## Appendix: Advanced Usage

### Customizing Generation Parameters

In the test script, you can adjust these parameters:

```python
audio, saved_path = generate_audio(
    model=model,
    text=test_text,
    output_path=output_path,
    speaker_id="S1",
    max_tokens=2000,      # Maximum tokens to generate (longer = more audio)
    cfg_scale=3.0,        # Classifier-free guidance scale (higher = more controlled)
    temperature=1.2,       # Sampling temperature (lower = more consistent)
    top_p=0.95,           # Nucleus sampling (controls diversity)
)
```

**Parameter Guide:**
- **max_tokens**: Increase for longer audio (try 1000-3000)
- **cfg_scale**: Higher = more faithful to text (try 2.0-5.0)
- **temperature**: Lower = more consistent (try 0.8-1.5)
- **top_p**: Controls diversity (try 0.9-0.99)

### Batch Processing Multiple Texts

Create a file `batch_generate.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "dia"))
from dia.model import Dia
import soundfile as sf
from datetime import datetime

# Your texts
texts = [
    "Hello, this is the first audio.",
    "This is the second audio with breaths 🌬️",
    "And this one has laughs 🤣"
]

# Load model (do this once)
model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", compute_dtype="float32", device="cpu")

# Generate audio for each text
for i, text in enumerate(texts):
    print(f"Generating audio {i+1}/{len(texts)}...")
    audio = model.generate(
        text=f"[S1] {text}",
        max_tokens=2000,
        cfg_scale=3.0,
        temperature=1.2,
        top_p=0.95,
    )
    
    # Save
    output_path = f"generated_audio/batch_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    sf.write(output_path, audio, 44100)
    print(f"Saved: {output_path}")
```

---

## Conclusion

Congratulations! You now know how to:
- ✅ Install and set up the Dia model
- ✅ Generate audio from text
- ✅ Use nonverbal tokens
- ✅ Troubleshoot common issues

**Remember:**
- First-time setup takes 20-30 minutes
- Model loading takes 2-5 minutes (first time only)
- Audio generation takes 30 seconds to 2 minutes
- Be patient and follow the steps carefully

**Happy audio generating! 🎵**

---

**Document Version:** 1.0  
**Last Updated:** November 2025  
**For:** Dia-1.6B NonverbalTTS Fine-tuned Model

