# Dia Text-to-Speech Training Project

This repository contains training scripts and tools for fine-tuning the [Dia-1.6B](https://huggingface.co/nari-labs/Dia-1.6B-0626) text-to-speech model on the NonverbalTTS dataset. The project includes both CPU and GPU training versions.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (for GPU version)
- Hugging Face account and access token

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/chakradharkalle03-arch/dia.git
   cd dia
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   
   # Install Dia package
   cd dia
   pip install -e .
   cd ..
   ```

4. **Set up Hugging Face token:**
   ```bash
   # Option 1: Use setup script
   python setup_hf_token.py
   
   # Option 2: Set environment variable manually
   # Windows
   set HF_TOKEN=your_token_here
   
   # Mac/Linux
   export HF_TOKEN=your_token_here
   ```

   Get your token from: https://huggingface.co/settings/tokens

### Quick Test

**CPU Version:**
```bash
cd CPU_version
python test_pretrained_model.py
```

**GPU Version:**
```bash
cd GPU_version
python test_pretrained_model_gpu.py
```

## 📁 Project Structure

```
dia/
├── CPU_version/          # CPU training scripts
│   ├── train_dia_proper.py
│   ├── test_pretrained_model.py
│   └── prepare_training_data.py
├── GPU_version/          # GPU training scripts
│   ├── train_dia_proper_gpu.py
│   ├── test_pretrained_model_gpu.py
│   └── prepare_training_data_gpu.py
├── dia/                  # Dia model package
├── processed_data/       # Processed dataset
└── requirements.txt      # Dependencies
```

## 🎯 Usage

### 1. Prepare Dataset

The scripts automatically download and process the NonverbalTTS dataset:

**CPU Version:**
```bash
python prepare_training_data.py
```

**GPU Version:**
```bash
cd GPU_version
python prepare_training_data_gpu.py
```

This will:
- Download the NonverbalTTS dataset from HuggingFace
- Convert emoji tokens to text tags (🌬️ → (breaths), 🤣 → (laughs), etc.)
- Format text with speaker tags [S1] and [S2]
- Save processed data to `processed_data/processed_dataset.json`

### 2. Train the Model

**CPU Version:**
```bash
cd CPU_version
python train_dia_proper.py
```

**GPU Version:**
```bash
cd GPU_version
python train_dia_proper_gpu.py
```

### 3. Test the Model

**CPU Version:**
```bash
cd CPU_version
python test_pretrained_model.py
```

**GPU Version:**
```bash
cd GPU_version
python test_pretrained_model_gpu.py
```

## 🔧 GPU Setup

If you're using the GPU version, verify CUDA is available:

```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
```

If CUDA is not available, install PyTorch with CUDA support:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 📊 Dataset Information

- **Dataset**: [NonverbalTTS](https://huggingface.co/datasets/deepvk/NonverbalTTS)
- **Size**: 17-hour English speech corpus
- **Features**: Nonverbal vocalizations (breaths, laughs, coughs, etc.)

### Nonverbal Token Mapping

| Emoji | Text Tag |
|-------|----------|
| 🌬️ | (breaths) |
| 😤 | (grunts) |
| 😷 | (coughs) |
| 👃 | (sniffs) |
| 🤣 | (laughs) |
| 🤧 | (sneezes) |
| 🗣️ | (vocal_noise) |
| 😖 | (groans) |
| 🐖 | (snorts) |

## 🎤 TTS Model Fine-tuning

This repository now includes scripts to fine-tune proper TTS base models on the NonverbalTTS dataset:

### Available Models

1. **SpeechT5** (Recommended) - Easy to fine-tune, good quality
   ```bash
   python train_speecht5.py
   ```

2. **XTTS-v2** (Advanced) - Excellent quality, supports voice cloning
   ```bash
   python train_xtts.py
   ```

### Quick Start for Fine-tuning

```bash
# 1. Install TTS dependencies
pip install -r requirements_tts.txt

# 2. Download audio dataset
python download_audio_dataset.py

# 3. Fine-tune SpeechT5 (easier option)
python train_speecht5.py --dataset dataset_audio/dataset_with_audio.json

# 4. Test fine-tuned model
python test_finetuned_model.py --checkpoint checkpoints/speecht5_finetuned/epoch_10
```

See **[TTS_FINETUNING_GUIDE.md](TTS_FINETUNING_GUIDE.md)** for detailed instructions.

## 📚 Documentation

- **[TTS_FINETUNING_GUIDE.md](TTS_FINETUNING_GUIDE.md)** - Complete guide for TTS fine-tuning
- **[QUICK_START.md](QUICK_START.md)** - Detailed command reference
- **[README_TRAINING.md](README_TRAINING.md)** - Training documentation
- **[USER_MANUAL.md](USER_MANUAL.md)** - User manual
- **[GPU_version/QUICKSTART.md](GPU_version/QUICKSTART.md)** - GPU-specific guide

## 🔗 Resources

- **Dia Model**: https://huggingface.co/nari-labs/Dia-1.6B-0626
- **Original Dia Repo**: https://github.com/nari-labs/dia
- **NonverbalTTS Dataset**: https://huggingface.co/datasets/deepvk/NonverbalTTS
- **Hugging Face Tokens**: https://huggingface.co/settings/tokens

## ⚙️ Configuration

### Environment Variables

- `HF_TOKEN` or `HUGGINGFACE_TOKEN`: Your Hugging Face access token

### Checkpoints

- CPU checkpoints: `CPU_version/checkpoints/dia-nonverbal-lora/`
- GPU checkpoints: `GPU_version/checkpoints/dia-nonverbal-lora-gpu/`

## 🛠️ Troubleshooting

**Token not found:**
```bash
# Verify token is set
python -c "import os; print('Token set:', 'HF_TOKEN' in os.environ)"
```

**CUDA not available:**
- Check GPU drivers are installed
- Verify PyTorch CUDA installation
- Use CPU version scripts if GPU is unavailable

**Dataset download issues:**
- Ensure HF_TOKEN is set correctly
- Check internet connection
- Verify HuggingFace account has dataset access

## 📝 Notes

- The model uses speaker tags `[S1]` and `[S2]` for dialogue
- Always alternate between speaker tags (don't use `[S1]` twice in a row)
- Nonverbal tokens should be used sparingly
- Generated audio files are saved in `generated_audio/` folders

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](dia/LICENSE) file for details.

## 🙏 Acknowledgments

- [Nari Labs](https://github.com/nari-labs) for the Dia model
- [NonverbalTTS Dataset](https://huggingface.co/datasets/deepvk/NonverbalTTS) creators

---

For more detailed information, see the documentation files in the repository.

