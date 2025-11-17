# ✅ Training Started Successfully!

## Current Status

**Training is now running in the background!**

### What's Running:
- **Script**: `CPU_version/train_dia_proper.py`
- **Model**: Dia-1.6B on NonverbalTTS dataset
- **Method**: Full fine-tuning with LoRA support (if available)
- **Process**: Running in background

## Monitor Training

### Check if Training is Running:
```powershell
Get-Process python | Select-Object Id, ProcessName, StartTime
```

### Check Training Progress:
```powershell
# View latest checkpoint
Get-ChildItem "CPU_version\checkpoints\dia-nonverbal-lora" -Directory | Sort-Object Name | Select-Object -Last 1

# View training history
Get-Content "CPU_version\checkpoints\dia-nonverbal-lora\epoch_10\training_history.json"
```

### Check Training Logs:
```powershell
Get-Content "training_log.txt" -Tail 20
```

## Training Configuration

- **Model**: Dia-1.6B-0626
- **Dataset**: NonverbalTTS (processed)
- **Epochs**: 50
- **Batch Size**: 2
- **Learning Rate**: 2e-4
- **Gradient Accumulation**: 4
- **Device**: CPU

## Checkpoints

Checkpoints are saved:
- Every 10 epochs
- When loss improves
- Final checkpoint at epoch 50

Location: `CPU_version/checkpoints/dia-nonverbal-lora/`

## Expected Timeline

- **Per Epoch**: ~2-4 hours (CPU)
- **Total Time**: ~20-40 hours for 10 epochs
- **Full Training**: ~100-200 hours for 50 epochs

## Next Steps

1. ✅ Training is running
2. ⏳ Wait for first checkpoint (epoch 10)
3. ⏳ Monitor progress periodically
4. ⏳ Test model after training completes

## Tips

- Training runs in background - you can close terminal
- Checkpoints are saved automatically
- Training can be stopped and resumed
- Use GPU version for faster training (when available)

## Troubleshooting

If training stops:
1. Check error messages in terminal
2. Verify dataset exists: `processed_data/processed_dataset.json`
3. Check disk space
4. Restart training: `python CPU_version/train_dia_proper.py`

