# Quick Start Guide - GPU Version

## Setup

1. **Verify GPU is detected:**
   ```bash
   python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
   ```

2. **If CUDA is False, install PyTorch with CUDA support:**
   ```bash
   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

## Training

```bash
cd GPU_version
python train_dia_proper_gpu.py
```

## Testing

```bash
cd GPU_version
python test_pretrained_model_gpu.py
```

## Important Notes

- GPU scripts will **error** if CUDA is not available (to prevent accidental CPU usage)
- Use CPU version scripts in parent directory if GPU is not available
- Checkpoints are saved separately: `checkpoints/dia-nonverbal-lora-gpu/`
- Generated audio files are prefixed with `gpu_` to distinguish from CPU outputs

