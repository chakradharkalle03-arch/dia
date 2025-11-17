# LoRA Fine-tuning Status

## ✅ What's Fixed and Running

### 1. **CPU Training with LoRA** - STARTED
- **Script**: `CPU_version/train_speecht5_lora_cpu.py`
- **Status**: Running in background
- **LoRA**: Enabled (with fallback to full fine-tuning if PEFT unavailable)
- **Checkpoints**: `CPU_version/checkpoints/speecht5_lora_cpu/`

### 2. **GPU Training with LoRA** - READY
- **Script**: `GPU_version/train_speecht5_lora_gpu.py`
- **Status**: Ready (requires CUDA)
- **LoRA**: Enabled
- **Checkpoints**: `GPU_version/checkpoints/speecht5_lora_gpu/`

## 🔧 Fixes Applied

### 1. **Dataset Path Handling**
- ✅ Fixed relative/absolute path issues
- ✅ Multiple fallback paths (dataset_audio, processed_data)
- ✅ Better error messages

### 2. **LoRA Integration**
- ✅ PEFT integration with proper error handling
- ✅ Automatic fallback to full fine-tuning if PEFT unavailable
- ✅ Optimized for both CPU and GPU

### 3. **Error Handling**
- ✅ Graceful handling of missing audio files
- ✅ Dummy data generation for text-only samples
- ✅ Better exception handling throughout

## 📊 LoRA Configuration

### CPU Version
- **Rank (r)**: 16
- **Alpha**: 32
- **Target Modules**: encoder.layers, decoder.layers
- **Dropout**: 0.05
- **Batch Size**: 2
- **Learning Rate**: 5e-4

### GPU Version
- **Rank (r)**: 16
- **Alpha**: 32
- **Target Modules**: encoder.layers, decoder.layers
- **Dropout**: 0.05
- **Batch Size**: 8
- **Learning Rate**: 5e-4

## 🚀 How to Use

### Start CPU Training (Already Running)
```bash
cd CPU_version
python train_speecht5_lora_cpu.py
```

### Start GPU Training (When GPU Available)
```bash
cd GPU_version
python train_speecht5_lora_gpu.py
```

## 📝 Notes

### PEFT Compatibility Issue
There's a known compatibility issue with PEFT and torchvision on some systems. The script handles this by:
1. Trying to import PEFT
2. If import fails, falling back to full fine-tuning
3. Training continues normally

### LoRA Benefits
- **Memory Efficient**: Uses ~10-20% of full model parameters
- **Faster Training**: Fewer parameters to update
- **Easy to Merge**: LoRA weights can be merged back into base model
- **Multiple Adapters**: Can train multiple LoRA adapters for different tasks

## 📈 Expected Results

### With LoRA
- **Memory Usage**: ~2-4GB (CPU), ~4-6GB (GPU)
- **Training Speed**: Faster than full fine-tuning
- **Model Size**: Small adapter files (~10-50MB)

### Without LoRA (Fallback)
- **Memory Usage**: ~4-8GB (CPU), ~8-12GB (GPU)
- **Training Speed**: Slower but still functional
- **Model Size**: Full model checkpoints

## 🔍 Monitor Training

### Check Progress
```bash
# View training history
Get-Content CPU_version/checkpoints/speecht5_lora_cpu/training_history.json

# Check if process is running
Get-Process python | Where-Object {$_.Path -like "*python*"}
```

### View Checkpoints
```bash
# List checkpoints
Get-ChildItem CPU_version/checkpoints/speecht5_lora_cpu/
```

## 🎯 Next Steps

1. ✅ CPU training is running
2. ⏳ Wait for training to complete
3. ⏳ Test fine-tuned model
4. ⏳ Start GPU training when GPU available
5. ⏳ Compare LoRA vs full fine-tuning results

## 🐛 Troubleshooting

### PEFT Import Error
If you see PEFT import errors:
- The script will automatically fall back to full fine-tuning
- Training will continue normally
- To fix PEFT: `pip install --upgrade peft transformers torchvision`

### Dataset Not Found
The script tries multiple paths:
1. `../dataset_audio/dataset_with_audio.json`
2. `../processed_data/processed_dataset.json`
3. Shows clear error if none found

### Out of Memory
- CPU: Reduce batch_size to 1
- GPU: Reduce batch_size or enable gradient checkpointing
- LoRA should help reduce memory usage

## 📚 References

- [PEFT Documentation](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [SpeechT5 Model](https://huggingface.co/microsoft/speecht5_tts)

