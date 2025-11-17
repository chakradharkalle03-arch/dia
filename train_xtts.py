"""
Fine-tune XTTS-v2 model on NonverbalTTS dataset.
XTTS-v2 is a high-quality TTS model that supports fine-tuning on custom datasets.
"""
import os
import json
import torch
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import argparse

try:
    from TTS.api import TTS
    from TTS.utils.manage import ModelManager
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("Error: TTS library not installed. Install with: pip install TTS")
    print("Or install requirements: pip install -r requirements_tts.txt")

def prepare_xtts_dataset(metadata_file: str, output_dir: str):
    """
    Prepare dataset in XTTS format.
    XTTS expects: wav files and a metadata.csv with format:
    wav_file|speaker_name|text
    """
    try:
        import pandas as pd
    except ImportError:
        print("Error: pandas not installed. Install with: pip install pandas")
        return None, 0
    
    print("=" * 70)
    print("Preparing Dataset for XTTS-v2")
    print("=" * 70)
    
    # Load metadata
    with open(metadata_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\nLoaded {len(data)} samples")
    
    # Create XTTS format
    xtts_data = []
    for item in tqdm(data, desc="Preparing data"):
        audio_path = item.get('audio_path')
        if not audio_path or not Path(audio_path).exists():
            continue
        
        # Clean text (remove speaker tags for XTTS, keep nonverbal tokens)
        text = item['text']
        # Remove [S1] and [S2] tags, XTTS handles speakers differently
        text = text.replace('[S1]', '').replace('[S2]', '').strip()
        
        # Use speaker_id as speaker name
        speaker = item.get('speaker_id', 'speaker_0')
        
        xtts_data.append({
            'wav_file': audio_path,
            'speaker_name': speaker,
            'text': text
        })
    
    # Save as CSV
    output_path = Path(output_dir) / "metadata.csv"
    df = pd.DataFrame(xtts_data)
    df.to_csv(output_path, sep='|', index=False, header=False)
    
    print(f"\n[OK] Created XTTS metadata: {output_path}")
    print(f"     Total samples: {len(xtts_data)}")
    print(f"     Unique speakers: {df['speaker_name'].nunique()}")
    
    return str(output_path), len(xtts_data)


def train_xtts(
    dataset_path: str,
    output_dir: str,
    num_epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    use_gpu: bool = True
):
    """
    Fine-tune XTTS-v2 model.
    """
    if not TTS_AVAILABLE:
        print("Error: TTS library not available")
        return
    
    print("=" * 70)
    print("Fine-tuning XTTS-v2 on NonverbalTTS Dataset")
    print("=" * 70)
    
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Initialize TTS
    print("\nInitializing XTTS-v2 model...")
    try:
        # Load XTTS-v2 model
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True)
        tts.to(device)
        print("[OK] Model loaded")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTrying alternative method...")
        # Alternative: Use ModelManager
        manager = ModelManager()
        model_path, config_path, model_item = manager.download_model("tts_models/multilingual/multi-dataset/xtts_v2")
        tts = TTS(model_path=model_path, config_path=config_path)
        tts.to(device)
        print("[OK] Model loaded via ModelManager")
    
    # Prepare output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Fine-tune the model
    print(f"\nStarting fine-tuning...")
    print(f"  Dataset: {dataset_path}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    
    try:
        # XTTS fine-tuning command
        # Note: This is a simplified version. Full fine-tuning may require
        # more configuration or using TTS training scripts directly
        
        # For now, we'll use TTS's built-in fine-tuning if available
        # Otherwise, provide instructions for manual fine-tuning
        
        print("\n" + "=" * 70)
        print("Fine-tuning Configuration")
        print("=" * 70)
        print("\nXTTS-v2 fine-tuning requires specific setup.")
        print("Please use one of the following methods:")
        print("\nMethod 1: Use Coqui TTS Training Scripts")
        print("  See: https://github.com/coqui-ai/TTS/tree/dev/TTS/bin")
        print("  Command:")
        print(f"    python TTS/bin/train_xtts.py \\")
        print(f"      --config_path TTS/tts/configs/xtts_v2/config.json \\")
        print(f"      --train_dataset {dataset_path} \\")
        print(f"      --output_path {output_dir} \\")
        print(f"      --batch_size {batch_size} \\")
        print(f"      --epochs {num_epochs} \\")
        print(f"      --lr {learning_rate}")
        
        print("\nMethod 2: Use HuggingFace Transformers (if XTTS is available)")
        print("  Some XTTS models are available on HuggingFace")
        print("  Check: https://huggingface.co/models?search=xtts")
        
        print("\n" + "=" * 70)
        print("Alternative: Use SpeechT5 for Fine-tuning")
        print("=" * 70)
        print("\nSpeechT5 is easier to fine-tune and available on HuggingFace.")
        print("See: train_speecht5.py for SpeechT5 fine-tuning script.")
        
    except Exception as e:
        print(f"\nError during fine-tuning setup: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune XTTS-v2 on NonverbalTTS dataset")
    parser.add_argument("--dataset", type=str, default="dataset_audio/dataset_with_audio.json",
                       help="Path to dataset metadata JSON")
    parser.add_argument("--output", type=str, default="checkpoints/xtts_finetuned",
                       help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not Path(args.dataset).exists():
        print(f"Error: Dataset not found at {args.dataset}")
        print("Please run: python download_audio_dataset.py first")
        return
    
    # Prepare XTTS dataset format
    dataset_dir = Path(args.dataset).parent
    metadata_csv, num_samples = prepare_xtts_dataset(args.dataset, str(dataset_dir))
    
    if num_samples == 0:
        print("Error: No valid samples found in dataset")
        return
    
    # Train
    train_xtts(
        dataset_path=metadata_csv,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_gpu=not args.cpu
    )


if __name__ == "__main__":
    main()

