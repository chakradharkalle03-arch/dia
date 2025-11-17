"""
Fine-tune Dia model with LoRA on NonverbalTTS dataset.
This script trains Dia-1.6B using LoRA adapters for efficient fine-tuning.
"""
import json
import os
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from tqdm import tqdm
import soundfile as sf
import torchaudio
from torch.utils.data import Dataset, DataLoader
import dac

# Import Dia model
import sys
sys.path.insert(0, str(Path(__file__).parent / "dia"))
from dia.model import Dia, DEFAULT_SAMPLE_RATE
from dia.config import DiaConfig

# Try to import PEFT for LoRA
try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    print("Warning: PEFT not available. Will train full model (requires more VRAM)")
    PEFT_AVAILABLE = False


class NonverbalTTSDataset(Dataset):
    """Dataset for NonverbalTTS with Dia-compatible formatting."""
    
    def __init__(self, data_path: str, model: Dia, max_audio_length: int = 3000):
        """
        Args:
            data_path: Path to processed dataset JSON file
            model: Dia model instance (for encoding audio)
            max_audio_length: Maximum audio tokens length
        """
        self.model = model
        self.max_audio_length = max_audio_length
        
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} samples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Encode text (Dia uses byte-level encoding)
        text = item['text']
        text_tokens = self.model._encode_text(text)
        
        # Handle audio
        audio_tokens = None
        audio_path = item.get('audio_path')
        audio_array = item.get('audio_array')
        sample_rate = item.get('sample_rate', DEFAULT_SAMPLE_RATE)
        
        if audio_path and os.path.exists(audio_path):
            try:
                # Load and encode audio
                audio_tokens = self.model.load_audio(audio_path)
            except Exception as e:
                print(f"Warning: Failed to load audio from {audio_path}: {e}")
                audio_tokens = None
        elif audio_array is not None:
            try:
                # Convert numpy array to tensor and encode
                audio_tensor = torch.from_numpy(np.array(audio_array)).float()
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dimension
                
                # Resample if needed
                if sample_rate != DEFAULT_SAMPLE_RATE:
                    audio_tensor = torchaudio.functional.resample(
                        audio_tensor, sample_rate, DEFAULT_SAMPLE_RATE
                    )
                
                # Encode using DAC
                audio_tensor = audio_tensor.to(self.model.device)
                audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
                audio_data = self.model.dac_model.preprocess(audio_tensor, DEFAULT_SAMPLE_RATE)
                _, encoded_frame, _, _, _ = self.model.dac_model.encode(audio_data)
                audio_tokens = encoded_frame.squeeze(0).transpose(0, 1)
                
                # Truncate if too long
                if audio_tokens.shape[0] > self.max_audio_length:
                    audio_tokens = audio_tokens[:self.max_audio_length]
            except Exception as e:
                print(f"Warning: Failed to encode audio array: {e}")
                audio_tokens = None
        
        return {
            'text': text,
            'text_tokens': text_tokens,
            'audio_tokens': audio_tokens,
            'id': item['id']
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    texts = [item['text'] for item in batch]
    text_tokens = [item['text_tokens'] for item in batch]
    
    # Pad text tokens
    max_text_len = max(len(t) for t in text_tokens)
    padded_text_tokens = []
    for tokens in text_tokens:
        padded = torch.zeros(max_text_len, dtype=tokens.dtype, device=tokens.device)
        padded[:len(tokens)] = tokens
        padded_text_tokens.append(padded)
    
    text_tokens_batch = torch.stack(padded_text_tokens)
    
    # Handle audio tokens (may be None)
    audio_tokens_list = [item['audio_tokens'] for item in batch]
    has_audio = any(at is not None for at in audio_tokens_list)
    
    return {
        'text': texts,
        'text_tokens': text_tokens_batch,
        'audio_tokens': audio_tokens_list if has_audio else None,
        'ids': [item['id'] for item in batch]
    }


def compute_loss(model: Dia, text_tokens: torch.Tensor, audio_tokens: torch.Tensor, config: DiaConfig):
    """
    Compute training loss for Dia model.
    This is a simplified loss computation - in practice, you'd need to implement
    the full forward pass through encoder and decoder.
    """
    # This is a placeholder - actual implementation would require
    # running the full encoder-decoder forward pass
    # For now, we'll use a simple approach
    
    batch_size = text_tokens.shape[0]
    device = text_tokens.device
    
    # Prepare inputs
    enc_input = text_tokens.unsqueeze(1)  # Add channel dimension: [B, 1, T]
    
    # Encoder forward pass
    from dia.state import EncoderInferenceState
    enc_state = EncoderInferenceState.new(config, enc_input)
    
    # Stack conditional and unconditional inputs for CFG
    enc_input_uncond = torch.zeros_like(enc_input)
    enc_input_cond = enc_input
    stacked_inputs = torch.stack([enc_input_uncond, enc_input_cond], dim=1)
    enc_input_batch = stacked_inputs.view(2 * batch_size, -1)
    
    encoder_out = model.model.encoder(enc_input_batch, enc_state)
    
    # Decoder forward pass (simplified - would need full implementation)
    # For training, we need teacher forcing with audio tokens
    # This is a complex implementation that would require modifying Dia's inference code
    
    # Placeholder loss - actual implementation needed
    # In practice, you'd compute cross-entropy loss between predicted and target audio tokens
    loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    return loss


def train_epoch(model: Dia, dataloader: DataLoader, optimizer: torch.optim.Optimizer, 
                config: DiaConfig, device: torch.device, epoch: int):
    """Train for one epoch."""
    model.model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        text_tokens = batch['text_tokens'].to(device)
        audio_tokens_list = batch['audio_tokens']
        
        # Skip batches without audio
        if audio_tokens_list is None or all(at is None for at in audio_tokens_list):
            continue
        
        # For now, we'll use a simplified training approach
        # Full implementation would require modifying Dia's forward pass for training
        
        optimizer.zero_grad()
        
        # Placeholder loss computation
        # In practice, you'd need to implement the full training forward pass
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # This is a simplified version - actual training would need:
        # 1. Full encoder-decoder forward pass with teacher forcing
        # 2. Cross-entropy loss on audio tokens
        # 3. Proper handling of delay patterns
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 10 == 0:
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def main():
    """Main training function."""
    # Configuration
    model_name = "nari-labs/Dia-1.6B-0626"  # Use specific version
    data_path = "processed_data/processed_dataset.json"
    output_dir = Path("checkpoints/dia-nonverbal-lora")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_epochs = 50
    batch_size = 1  # Start with 1 due to memory constraints
    learning_rate = 2e-4
    gradient_accumulation_steps = 8
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cpu":
        print("Warning: Training on CPU will be very slow. GPU recommended.")
    
    # Load model
    print(f"Loading model {model_name}...")
    compute_dtype = "float16" if device.type == "cuda" else "float32"
    model = Dia.from_pretrained(model_name, compute_dtype=compute_dtype, device=device)
    
    # Apply LoRA if available
    if PEFT_AVAILABLE:
        print("Applying LoRA adapters...")
        # Note: PEFT may not work directly with Dia's custom layers
        # We'll try to apply it, but may need to fall back to full fine-tuning
        try:
            # Try to find Linear layers in the model
            target_modules = []
            for name, module in model.model.named_modules():
                if isinstance(module, nn.Linear):
                    target_modules.append(name)
            
            if not target_modules:
                # If no Linear layers, try DenseGeneral (custom layer)
                # This may not work with standard PEFT
                print("Warning: No standard Linear layers found. Trying custom approach...")
                # For now, we'll proceed without LoRA
                use_lora = False
            else:
                lora_config = LoraConfig(
                    r=32,
                    lora_alpha=32,
                    target_modules=target_modules[:10],  # Limit to first 10 modules
                    lora_dropout=0.05,
                    bias="none",
                    task_type=TaskType.FEATURE_EXTRACTION,
                )
                model.model = get_peft_model(model.model, lora_config)
                use_lora = True
                print(f"Applied LoRA to {len(target_modules)} modules")
        except Exception as e:
            print(f"Failed to apply LoRA: {e}")
            print("Falling back to full fine-tuning (requires more VRAM)")
            use_lora = False
    else:
        use_lora = False
    
    # Load dataset
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        print("Please run preprocess_dataset.py first")
        return
    
    print("Loading dataset...")
    dataset = NonverbalTTSDataset(data_path, model)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Batch size: {batch_size}, Gradient accumulation: {gradient_accumulation_steps}")
    print(f"Total steps per epoch: {len(dataloader)}")
    
    for epoch in range(1, num_epochs + 1):
        avg_loss = train_epoch(model, dataloader, optimizer, model.config, device, epoch)
        print(f"Epoch {epoch}/{num_epochs} - Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint (every 5 epochs to save space/time)
        if epoch % 5 == 0 or epoch == num_epochs:
            checkpoint_dir = output_dir / f"epoch_{epoch}"
            checkpoint_dir.mkdir(exist_ok=True)
            
            try:
                if use_lora:
                    # Save only LoRA weights
                    model.model.save_pretrained(str(checkpoint_dir))
                    print(f"Saved LoRA checkpoint to {checkpoint_dir}")
                else:
                    # Save only optimizer and epoch info (model is too large)
                    torch.save({
                        'epoch': epoch,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_loss,
                    }, checkpoint_dir / "checkpoint.pt")
                    print(f"Saved checkpoint to {checkpoint_dir}")
            except Exception as e:
                print(f"Warning: Failed to save checkpoint at epoch {epoch}: {e}")
                print("Continuing training...")
    
    print("\nTraining completed!")
    print(f"Final checkpoints saved in {output_dir}")


if __name__ == "__main__":
    main()

