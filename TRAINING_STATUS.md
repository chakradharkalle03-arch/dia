# Fine-tuning Status

## Current Status

✅ **CPU Training**: Started in background
⏳ **GPU Training**: Waiting (CUDA not available on this system)
⏳ **Audio Dataset**: Downloading in background

## Training Scripts Created

### CPU Version
- **Script**: `CPU_version/train_speecht5_cpu.py`
- **Status**: Running
- **Checkpoints**: `CPU_version/checkpoints/speecht5_finetuned_cpu/`
- **Configuration**:
  - Batch size: 2
  - Gradient accumulation: 8
  - Epochs: 10
  - Learning rate: 5e-5

### GPU Version
- **Script**: `GPU_version/train_speecht5_gpu.py`
- **Status**: Ready (requires CUDA)
- **Checkpoints**: `GPU_version/checkpoints/speecht5_finetuned_gpu/`
- **Configuration**:
  - Batch size: 8
  - Gradient accumulation: 2
  - Epochs: 10
  - Learning rate: 5e-5

## How to Monitor Training

### Check CPU Training Progress
```bash
# View training logs
Get-Content CPU_version/checkpoints/speecht5_finetuned_cpu/training_history.json

# Check if process is running
Get-Process python | Where-Object {$_.Path -like "*python*"}
```

### Start GPU Training (when GPU available)
```bash
cd GPU_version
python train_speecht5_gpu.py
```

### Use the Starter Script
```bash
python start_finetuning.py
```

This will:
1. Check requirements
2. Let you choose CPU, GPU, or both
3. Start training automatically

## Expected Training Time

- **CPU**: ~2-4 hours per epoch (10 epochs = 20-40 hours total)
- **GPU**: ~10-20 minutes per epoch (10 epochs = 2-3 hours total)

## Checkpoints

Checkpoints are saved every 5 epochs and when loss improves:
- `epoch_5/` - After 5 epochs
- `epoch_10/` - Final checkpoint
- Best model (lowest loss) is also saved

## Testing Fine-tuned Models

After training completes:

```bash
# Test CPU model
python test_finetuned_model.py \
    --checkpoint CPU_version/checkpoints/speecht5_finetuned_cpu/epoch_10 \
    --model_type speecht5 \
    --text "[S1] Hello! This is a test. (laughs) Can you hear the nonverbal sounds?" \
    --output test_cpu_output.wav

# Test GPU model (when available)
python test_finetuned_model.py \
    --checkpoint GPU_version/checkpoints/speecht5_finetuned_gpu/epoch_10 \
    --model_type speecht5 \
    --text "[S1] Hello! This is a test. (laughs)" \
    --output test_gpu_output.wav
```

## Troubleshooting

### Training stops or errors
1. Check if audio dataset is fully downloaded
2. Verify HuggingFace token is set
3. Check available disk space
4. Review error logs in checkpoint directories

### Out of memory
- CPU: Reduce batch_size in script (currently 2)
- GPU: Reduce batch_size or enable gradient checkpointing

### Slow training
- CPU training is inherently slow - consider using GPU if available
- Reduce number of epochs for testing
- Use smaller dataset subset for faster iteration

## Next Steps

1. Wait for training to complete
2. Evaluate checkpoints
3. Test best model
4. Compare CPU vs GPU results (when GPU available)
5. Fine-tune hyperparameters if needed

