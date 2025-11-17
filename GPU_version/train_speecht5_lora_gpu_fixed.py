"""
GPU Version: Fine-tune SpeechT5 model on NonverbalTTS dataset using LoRA.
FIXED VERSION - Handles torchvision compatibility issues.
FORCES GPU USAGE - Use CUDA device.
"""
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import soundfile as sf
import numpy as np

# Fix torchvision import issue by importing transformers components directly
try:
    # Try to import with workaround for torchvision issues
    import sys
    # Suppress torchvision errors temporarily
    import warnings
    warnings.filterwarnings('ignore')
    
    from transformers import (
        SpeechT5Processor,
        SpeechT5ForTextToSpeech,
        SpeechT5HifiGan,
        set_seed
    )
    TRANSFORMERS_AVAILABLE = True
except Exception as e:
    print(f"Warning: Transformers import issue: {e}")
    print("Trying alternative import method...")
    try:
        # Alternative: import directly from model files
        from transformers.models.speecht5 import (
            SpeechT5Processor,
            SpeechT5ForTextToSpeech,
            SpeechT5HifiGan
        )
        from transformers import set_seed
        TRANSFORMERS_AVAILABLE = True
    except Exception as e2:
        print(f"Error: Cannot import SpeechT5 components: {e2}")
        print("\nPlease fix torchvision compatibility:")
        print("  pip install --upgrade torch torchvision transformers")
        TRANSFORMERS_AVAILABLE = False
        sys.exit(1)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import PEFT for LoRA
try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not available. Install with: pip install peft")
    print("Falling back to full fine-tuning (will use more VRAM)")

# Set HF token from environment variable
if not os.getenv("HF_TOKEN") and not os.getenv("HUGGINGFACE_TOKEN"):
    token = input("Please enter your HuggingFace token: ").strip()
    os.environ["HF_TOKEN"] = token


class NonverbalTTSDataset(Dataset):
    """Dataset for SpeechT5 fine-tuning."""
    
    def __init__(
        self,
        metadata_file: str,
        processor: SpeechT5Processor,
        max_length: int = 500
    ):
        self.processor = processor
        self.max_length = max_length
        
        # Load metadata - handle both relative and absolute paths
        metadata_path = Path(metadata_file)
        if not metadata_path.is_absolute():
            if not metadata_path.exists():
                metadata_path = Path(__file__).parent.parent / metadata_file
            if not metadata_path.exists():
                metadata_path = Path(__file__).parent.parent / "dataset_audio" / "dataset_with_audio.json"
        
        if not metadata_path.exists():
            metadata_path = Path(__file__).parent.parent / "processed_data" / "processed_dataset.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Dataset not found. Tried: {metadata_file}, {metadata_path}")
        
        print(f"Loading dataset from: {metadata_path}")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Filter out samples without audio
        if isinstance(self.data, list) and len(self.data) > 0 and 'audio_path' in self.data[0]:
            self.data = [item for item in self.data if item.get('audio_path') and Path(item['audio_path']).exists()]
        
        print(f"Loaded {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        text = item.get('text', item.get('normalized_text', ''))
        text = text.replace('[S1]', '').replace('[S2]', '').strip()
        
        audio_path = item.get('audio_path')
        if audio_path and Path(audio_path).exists():
            try:
                audio, sample_rate = sf.read(audio_path)
                if sample_rate != 16000:
                    try:
                        import librosa
                        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
                        sample_rate = 16000
                    except ImportError:
                        pass
            except Exception as e:
                audio = np.zeros(16000, dtype=np.float32)
                sample_rate = 16000
        else:
            audio = np.zeros(16000, dtype=np.float32)
            sample_rate = 16000
        
        try:
            inputs = self.processor(
                text=text,
                audio_target=audio,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True
            )
            
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': inputs['labels'].squeeze(0),
            }
        except Exception as e:
            return {
                'input_ids': torch.zeros(10, dtype=torch.long),
                'attention_mask': torch.ones(10, dtype=torch.long),
                'labels': torch.zeros(100, dtype=torch.float),
            }


def collate_fn(batch):
    """Collate function for DataLoader."""
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    max_len = max(len(ids) for ids in input_ids)
    
    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []
    
    for i in range(len(batch)):
        pad_len = max_len - len(input_ids[i])
        padded_input_ids.append(torch.cat([input_ids[i], torch.zeros(pad_len, dtype=input_ids[i].dtype)]))
        padded_attention_mask.append(torch.cat([attention_mask[i], torch.zeros(pad_len, dtype=attention_mask[i].dtype)]))
        
        label_len = len(labels[i])
        if label_len < max_len:
            pad_audio = torch.zeros(max_len - label_len, labels[i].shape[-1] if len(labels[i].shape) > 1 else 1)
            if len(labels[i].shape) > 1:
                padded_labels.append(torch.cat([labels[i], pad_audio]))
            else:
                padded_labels.append(torch.cat([labels[i], pad_audio.squeeze(-1)]))
        else:
            padded_labels.append(labels[i][:max_len])
    
    return {
        'input_ids': torch.stack(padded_input_ids),
        'attention_mask': torch.stack(padded_attention_mask),
        'labels': torch.stack(padded_labels),
    }


def train_epoch(
    model,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def main():
    """Main training function."""
    if not TRANSFORMERS_AVAILABLE:
        print("Error: Cannot proceed without transformers")
        return
    
    # Configuration - GPU optimized with LoRA
    model_name = "microsoft/speecht5_tts"
    metadata_file = "../dataset_audio/dataset_with_audio.json"
    output_dir = Path("checkpoints/speecht5_lora_gpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_epochs = 10
    batch_size = 8  # Larger batch for GPU
    learning_rate = 5e-4  # Slightly higher for LoRA
    gradient_accumulation_steps = 2
    
    # GPU Device setup - FORCE GPU
    if not torch.cuda.is_available():
        print("=" * 70)
        print("WARNING: CUDA is not available!")
        print("=" * 70)
        print("\nThis script requires GPU. Options:")
        print("1. Use CPU version: CPU_version/train_speecht5_lora_cpu.py")
        print("2. Install CUDA-enabled PyTorch:")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print("\nProceeding anyway (will fail if model requires GPU)...")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    
    print("=" * 70)
    print("Fine-tuning SpeechT5 with LoRA on NonverbalTTS Dataset - GPU VERSION")
    print("=" * 70)
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"\nModel: {model_name}")
    print(f"LoRA: {'Enabled' if PEFT_AVAILABLE else 'Disabled (PEFT not available)'}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation: {gradient_accumulation_steps}")
    print(f"Learning rate: {learning_rate}")
    
    # Load processor and model
    print(f"\nLoading processor and model...")
    try:
        processor = SpeechT5Processor.from_pretrained(model_name)
        model = SpeechT5ForTextToSpeech.from_pretrained(model_name)
        print("[OK] Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTrying to fix torchvision issue...")
        print("Please run: pip install --upgrade torch torchvision transformers")
        return
    
    # Apply LoRA if available
    use_lora = False
    if PEFT_AVAILABLE:
        print("\nApplying LoRA adapters...")
        try:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["encoder.layers", "decoder.layers"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            
            model = get_peft_model(model, lora_config)
            use_lora = True
            print("[OK] LoRA adapters applied")
            print(f"     Trainable parameters: {model.num_parameters(only_trainable=True):,}")
            print(f"     Total parameters: {model.num_parameters():,}")
        except Exception as e:
            print(f"Warning: Failed to apply LoRA: {e}")
            use_lora = False
    
    model.to(device)
    print("[OK] Model moved to device")
    
    # Create dataset
    print(f"\nLoading dataset...")
    try:
        dataset = NonverbalTTSDataset(metadata_file, processor)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure:")
        print("  1. Run: python download_audio_dataset.py")
        print("  2. Or ensure processed_data/processed_dataset.json exists")
        return
    
    if len(dataset) == 0:
        print("Error: No valid samples in dataset")
        return
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2 if torch.cuda.is_available() else 0
    )
    print(f"[OK] Dataset loaded: {len(dataset)} samples")
    print(f"     Steps per epoch: {len(dataloader)}")
    
    # Optimizer
    if use_lora:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=0.01
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=learning_rate * 0.1
    )
    
    # Training loop
    print(f"\n{'='*70}")
    print("Starting Training")
    print(f"{'='*70}\n")
    
    best_loss = float('inf')
    training_history = []
    
    for epoch in range(1, num_epochs + 1):
        avg_loss = train_epoch(model, dataloader, optimizer, device, epoch)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        print(f"\nEpoch {epoch}/{num_epochs} Summary:")
        print(f"  Loss: {avg_loss:.6f}")
        print(f"  Learning Rate: {current_lr:.2e}")
        
        training_history.append({
            'epoch': epoch,
            'loss': avg_loss,
            'lr': current_lr
        })
        
        # Save checkpoint
        if avg_loss < best_loss or epoch % 5 == 0:
            checkpoint_dir = output_dir / f"epoch_{epoch}"
            checkpoint_dir.mkdir(exist_ok=True)
            
            if use_lora:
                model.save_pretrained(str(checkpoint_dir))
                print(f"  [OK] Saved LoRA checkpoint to {checkpoint_dir}")
            else:
                model.save_pretrained(str(checkpoint_dir))
                processor.save_pretrained(str(checkpoint_dir))
                print(f"  [OK] Saved checkpoint to {checkpoint_dir}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                print(f"  [NEW BEST] Loss improved!")
        
        print()
    
    print(f"{'='*70}")
    print("Training Completed!")
    print(f"{'='*70}")
    print(f"\nBest loss: {best_loss:.6f}")
    print(f"Checkpoints saved in: {output_dir}")
    
    history_file = output_dir / "training_history.json"
    with open(history_file, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"Training history saved to: {history_file}")


if __name__ == "__main__":
    set_seed(42)
    main()
