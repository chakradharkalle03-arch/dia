# Dia Model Fine-tuning on NonverbalTTS Dataset

This repository contains scripts to fine-tune the Dia-1.6B model on the NonverbalTTS dataset for improved nonverbal vocalization generation.

## Overview

- **Model**: Dia-1.6B (Nari Labs) - A 1.6B parameter dialogue-first TTS model
- **Dataset**: NonverbalTTS - 17-hour English speech corpus with nonverbal vocalizations
- **Goal**: Fine-tune Dia to better handle nonverbal tokens like (breaths), (laughs), (coughs), etc.

## Files

- `preprocess_dataset.py` - Preprocesses NonverbalTTS dataset for Dia format
- `train_dia_lora.py` - Full training script (framework - needs implementation)
- `train_simple.py` - Simplified setup script to verify environment
- `requirements_training.txt` - Additional dependencies for training

## Setup

1. **Install Dia**:
```bash
cd dia
pip install -e .
cd ..
```

2. **Install training dependencies**:
```bash
pip install datasets accelerate peft bitsandbytes
```

3. **Note**: There may be NumPy version conflicts. If you encounter issues:
```bash
pip install "numpy<2.0"
```

## Usage

### Step 1: Preprocess Dataset

```bash
python preprocess_dataset.py
```

This will:
- Load NonverbalTTS dataset from HuggingFace
- Convert emoji tokens (🌬️, 🤣, etc.) to text tags ((breaths), (laughs), etc.)
- Format text in Dia's expected format with speaker tags [S1], [S2]
- Save processed data to `processed_data/processed_dataset.json`

**Note**: You may encounter FFmpeg/torchcodec issues when loading audio. The preprocessing script handles this by accessing metadata only.

### Step 2: Verify Setup

```bash
python train_simple.py
```

This will:
- Load the Dia model
- Load the processed dataset
- Display configuration and sample data
- Verify everything is set up correctly

### Step 3: Training

**Important**: Full training implementation requires modifying Dia's architecture for training mode. The current Dia model is inference-only.

To implement training, you would need to:

1. **Add training forward pass**:
   - Implement teacher forcing for decoder
   - Handle delay patterns during training
   - Support batch processing with variable lengths

2. **Implement loss computation**:
   - Cross-entropy loss on audio tokens
   - Handle padding and masking
   - Support multiple channels

3. **Add LoRA support** (optional):
   - Dia uses custom `DenseGeneral` layers
   - Standard PEFT may not work directly
   - Consider custom LoRA implementation or full fine-tuning

## Current Status

✅ Dataset preprocessing framework  
✅ Model loading and setup  
✅ Configuration management  
⏳ Full training loop (requires Dia architecture modifications)  
⏳ LoRA adapter implementation  
⏳ Loss computation  

## Dataset Format

The processed dataset contains:
- `text`: Formatted text with speaker tags and nonverbal tokens
- `original_text`: Original text from dataset
- `audio_path`: Path to audio file (if available)
- `emotion`: Emotion label
- `speaker_id`: Speaker identifier
- `duration`: Audio duration in seconds

Example:
```json
{
  "text": "[S1] Yeah, it was a good time. (breaths) (grunts)",
  "original_text": "Yeah, it was a good time. 🌬️ 😤",
  "emotion": "happy",
  "speaker_id": "ex01"
}
```

## Nonverbal Token Mapping

| Emoji | Text Tag |
|-------|----------|
| 🌬️ | (breaths) |
| 😤 | (grunts) |
| 😷 | (coughs) |
| 👃 | (sniffs) |
| 🤣 | (laughs) |
| 🤧 | (sneezes) |
| 🗣️ | (vocal_noise) |

## Hardware Requirements

- **Minimum**: 12GB VRAM (with LoRA/QLoRA)
- **Recommended**: 16GB+ VRAM
- **Full fine-tuning**: 24GB+ VRAM

## Notes

- Dia uses byte-level text encoding (not a traditional tokenizer)
- Audio is encoded using Descript Audio Codec (DAC)
- The model uses a delay pattern for multi-channel audio generation
- Training requires significant modifications to Dia's inference-only architecture

## Next Steps

1. Implement training forward pass in `dia/model.py`
2. Add loss computation function
3. Implement proper batching and data loading
4. Add checkpointing and evaluation
5. Test on small subset first

## References

- [Dia Model](https://huggingface.co/nari-labs/Dia-1.6B)
- [NonverbalTTS Dataset](https://huggingface.co/datasets/deepvk/NonverbalTTS)
- [Dia Repository](https://github.com/nari-labs/dia)

