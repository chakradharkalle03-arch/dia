# GPU Version - Dia Model Training and Testing

This folder contains GPU-optimized versions of the training and testing scripts for the Dia model fine-tuned on the NonverbalTTS dataset.

## Requirements

- CUDA-capable GPU (NVIDIA GPU with CUDA support)
- CUDA installed and properly configured
- PyTorch with CUDA support
- All dependencies from `requirements_training.txt`

## Files

- `train_dia_proper_gpu.py` - GPU training script (forces CUDA usage)
- `test_pretrained_model_gpu.py` - GPU testing script (forces CUDA usage)
- `prepare_training_data_gpu.py` - Dataset preparation script (can use shared processed_data folder)

## Usage

### 1. Prepare Dataset (if not already done)

The dataset can be prepared using either the CPU or GPU version. The processed data is shared in the parent `processed_data` folder.

```bash
cd GPU_version
python prepare_training_data_gpu.py
```

Or use the CPU version (they share the same output):
```bash
python prepare_training_data.py
```

### 2. Train Model

```bash
cd GPU_version
python train_dia_proper_gpu.py
```

This will:
- Force GPU usage (will error if CUDA is not available)
- Train for 50 epochs
- Save checkpoints every 10 epochs in `checkpoints/dia-nonverbal-lora-gpu/`
- Use float16 precision for faster training

### 3. Test Model

```bash
cd GPU_version
python test_pretrained_model_gpu.py
```

This will:
- Force GPU usage
- Load the trained model from checkpoints
- Generate audio from test text
- Save output to `generated_audio/test_output_gpu_*.wav`

## Differences from CPU Version

1. **Device**: Forces CUDA usage (will error if GPU not available)
2. **Precision**: Uses float16 instead of float32 for faster training
3. **Checkpoints**: Saved in `checkpoints/dia-nonverbal-lora-gpu/` (separate from CPU version)
4. **Output**: Generated audio files prefixed with `gpu_` to distinguish from CPU outputs

## Notes

- The CPU version scripts remain unchanged in the parent directory
- Both versions can share the same `processed_data` folder
- Checkpoints are stored separately to avoid conflicts
- GPU version will be significantly faster for training and inference

