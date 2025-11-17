# TTS Model Fine-tuning Guide

This guide explains how to fine-tune TTS (Text-to-Speech) models on the NonverbalTTS dataset.

## Available Models

### 1. SpeechT5 (Recommended for Beginners)
- **Model**: `microsoft/speecht5_tts`
- **Pros**: Easy to fine-tune, well-documented, good quality
- **Cons**: May not handle nonverbal sounds as well as specialized models
- **Script**: `train_speecht5.py`

### 2. XTTS-v2 (Advanced)
- **Model**: `tts_models/multilingual/multi-dataset/xtts_v2`
- **Pros**: Excellent quality, supports voice cloning, multilingual
- **Cons**: More complex setup, requires Coqui TTS framework
- **Script**: `train_xtts.py`

## Quick Start

### Step 1: Install Dependencies

```bash
# Install TTS dependencies
pip install -r requirements_tts.txt

# Or install individually:
pip install TTS transformers datasets soundfile librosa
```

### Step 2: Download Audio Dataset

The NonverbalTTS dataset metadata is already processed, but we need to download the actual audio files:

```bash
python download_audio_dataset.py
```

This will:
- Download audio files from HuggingFace
- Save them to `dataset_audio/` directory
- Create `dataset_audio/dataset_with_audio.json` with audio paths

**Note**: This may take a while depending on your internet connection (17 hours of audio).

### Step 3: Fine-tune SpeechT5 (Easier Option)

```bash
python train_speecht5.py \
    --dataset dataset_audio/dataset_with_audio.json \
    --output checkpoints/speecht5_finetuned \
    --epochs 10 \
    --batch_size 4 \
    --lr 5e-5
```

### Step 4: Fine-tune XTTS-v2 (Advanced Option)

```bash
# First, prepare XTTS format
python train_xtts.py --dataset dataset_audio/dataset_with_audio.json

# Then follow instructions in the output for full training
# Or use Coqui TTS training scripts directly
```

### Step 5: Test Fine-tuned Model

```bash
# Test SpeechT5
python test_finetuned_model.py \
    --checkpoint checkpoints/speecht5_finetuned/epoch_10 \
    --model_type speecht5 \
    --text "[S1] Hello! This is a test. (laughs) Can you hear the nonverbal sounds?" \
    --output test_output.wav

# Test XTTS
python test_finetuned_model.py \
    --checkpoint checkpoints/xtts_finetuned \
    --model_type xtts \
    --text "[S1] Hello! This is a test. (laughs)" \
    --output test_output.wav
```

## Detailed Instructions

### SpeechT5 Fine-tuning

SpeechT5 is a versatile TTS model from Microsoft that's relatively easy to fine-tune.

**Advantages:**
- Simple API
- Good documentation
- Works well with HuggingFace ecosystem
- Reasonable quality

**Training Parameters:**
- `--epochs`: Number of training epochs (start with 10)
- `--batch_size`: Batch size (4-8 depending on GPU memory)
- `--lr`: Learning rate (5e-5 is a good starting point)

**Example:**
```bash
python train_speecht5.py \
    --dataset dataset_audio/dataset_with_audio.json \
    --output checkpoints/speecht5_finetuned \
    --epochs 10 \
    --batch_size 4 \
    --lr 5e-5
```

### XTTS-v2 Fine-tuning

XTTS-v2 is a high-quality TTS model from Coqui that supports voice cloning and multilingual synthesis.

**Advantages:**
- Excellent voice quality
- Supports voice cloning
- Multilingual support
- Good for expressive speech

**Setup:**
1. Install Coqui TTS: `pip install TTS`
2. Download the model (automatic on first use)
3. Use training scripts from Coqui TTS repository

**Note**: Full XTTS fine-tuning requires using Coqui's training scripts. See `train_xtts.py` for instructions.

## Dataset Format

The processed dataset contains:
- `text`: Formatted text with nonverbal tokens like `(laughs)`, `(breaths)`, etc.
- `audio_path`: Path to audio file (WAV format, 16kHz)
- `speaker_id`: Speaker identifier
- `emotion`: Emotion label
- `duration`: Audio duration in seconds

## Nonverbal Tokens

The dataset includes these nonverbal tokens:
- `(breaths)` - Breathing sounds
- `(laughs)` - Laughter
- `(coughs)` - Coughing
- `(sniffs)` - Sniffing
- `(sneezes)` - Sneezing
- `(groans)` - Groaning
- `(grunts)` - Grunting
- `(vocal_noise)` - Other vocal noises
- `(snorts)` - Snorting

## Hardware Requirements

### SpeechT5
- **Minimum**: 8GB VRAM
- **Recommended**: 16GB+ VRAM
- **CPU**: Possible but very slow

### XTTS-v2
- **Minimum**: 12GB VRAM
- **Recommended**: 24GB+ VRAM
- **CPU**: Not recommended

## Troubleshooting

### Out of Memory
- Reduce `batch_size`
- Use gradient accumulation
- Enable mixed precision training

### Audio Download Issues
- Check HuggingFace token is set: `export HF_TOKEN=your_token`
- Verify internet connection
- Try downloading in smaller batches

### Model Loading Errors
- Ensure all dependencies are installed
- Check model paths are correct
- Verify HuggingFace token has access

### Poor Quality Output
- Train for more epochs
- Adjust learning rate
- Check dataset quality
- Verify audio files are valid

## Next Steps

After fine-tuning:
1. Evaluate on test set
2. Compare with baseline model
3. Adjust hyperparameters if needed
4. Export model for deployment

## References

- [SpeechT5 Documentation](https://huggingface.co/docs/transformers/model_doc/speecht5)
- [Coqui TTS Documentation](https://docs.coqui.ai/)
- [XTTS-v2 Model](https://huggingface.co/coqui/XTTS-v2)
- [NonverbalTTS Dataset](https://huggingface.co/datasets/deepvk/NonverbalTTS)

