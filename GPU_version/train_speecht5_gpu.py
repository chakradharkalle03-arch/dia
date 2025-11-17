"""
GPU Version: Fine-tune SpeechT5 model on NonverbalTTS dataset.
Optimized for GPU training with larger batch sizes and faster processing.
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
from transformers import (
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    set_seed
)
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
        
        # Load metadata
        with open(metadata_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Filter out samples without audio
        self.data = [item for item in self.data if item.get('audio_path') and Path(item['audio_path']).exists()]
        
        print(f"Loaded {len(self.data)} samples with audio")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load audio
        audio_path = item['audio_path']
        try:
            audio, sample_rate = sf.read(audio_path)
        except Exception as e:
            print(f"Warning: Error loading {audio_path}: {e}")
            audio = np.zeros(16000, dtype=np.float32)
            sample_rate = 16000
        
        # Resample if needed (SpeechT5 expects 16kHz)
        if sample_rate != 16000:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
            except ImportError:
                print("Warning: librosa not available, skipping resampling")
        
        # Clean text (remove speaker tags, keep nonverbal tokens)
        text = item['text']
        text = text.replace('[S1]', '').replace('[S2]', '').strip()
        
        # Process with SpeechT5 processor
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
            print(f"Warning: Error processing sample {idx}: {e}")
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
    
    # Pad sequences
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
    model: SpeechT5ForTextToSpeech,
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
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        
        # Backward pass
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
    # Configuration - GPU optimized
    model_name = "microsoft/speecht5_tts"
    vocoder_name = "microsoft/speecht5_hifigan"
    metadata_file = "../dataset_audio/dataset_with_audio.json"
    output_dir = Path("checkpoints/speecht5_finetuned_gpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_epochs = 10
    batch_size = 8  # Larger batch for GPU
    learning_rate = 5e-5
    gradient_accumulation_steps = 2
    
    # GPU Device setup - FORCE GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! This script requires GPU. Please use CPU version instead.")
    
    device = torch.device("cuda")
    print("=" * 70)
    print("Fine-tuning SpeechT5 on NonverbalTTS Dataset - GPU VERSION")
    print("=" * 70)
    print(f"\nDevice: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"\nModel: {model_name}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation: {gradient_accumulation_steps}")
    print(f"Learning rate: {learning_rate}")
    
    # Check dataset
    if not Path(metadata_file).exists():
        print(f"\nError: Dataset not found at {metadata_file}")
        print("Please run: python download_audio_dataset.py first")
        return
    
    # Load processor and model
    print(f"\nLoading processor and model...")
    processor = SpeechT5Processor.from_pretrained(model_name)
    model = SpeechT5ForTextToSpeech.from_pretrained(model_name)
    model.to(device)
    print("[OK] Model loaded")
    
    # Load vocoder (for inference, not training)
    print("Loading vocoder...")
    vocoder = SpeechT5HifiGan.from_pretrained(vocoder_name)
    vocoder.to(device)
    print("[OK] Vocoder loaded")
    
    # Create dataset
    print(f"\nLoading dataset from {metadata_file}...")
    dataset = NonverbalTTSDataset(metadata_file, processor)
    
    if len(dataset) == 0:
        print("Error: No valid samples in dataset")
        return
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2  # GPU: can use multiple workers
    )
    print(f"[OK] Dataset loaded: {len(dataset)} samples")
    print(f"     Steps per epoch: {len(dataloader)}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    # Learning rate scheduler
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
    
    # Save training history
    history_file = output_dir / "training_history.json"
    with open(history_file, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"Training history saved to: {history_file}")


if __name__ == "__main__":
    set_seed(42)
    main()

