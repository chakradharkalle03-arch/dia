"""
Download audio files from NonverbalTTS dataset for TTS fine-tuning.
This script downloads the actual audio files needed for training.
"""
import os
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import soundfile as sf

# Set HF token from environment variable
if not os.getenv("HF_TOKEN") and not os.getenv("HUGGINGFACE_TOKEN"):
    token = input("Please enter your HuggingFace token: ").strip()
    os.environ["HF_TOKEN"] = token

print("=" * 70)
print("Downloading NonverbalTTS Audio Dataset")
print("=" * 70)

# Configuration
output_dir = Path("dataset_audio")
output_dir.mkdir(exist_ok=True)

metadata_file = Path("processed_data/processed_dataset.json")
if not metadata_file.exists():
    print(f"\nError: Processed dataset not found at {metadata_file}")
    print("Please run: python prepare_training_data.py first")
    exit(1)

# Load processed metadata
print(f"\nLoading metadata from {metadata_file}...")
with open(metadata_file, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

print(f"Found {len(metadata)} samples in metadata")

# Load dataset from HuggingFace
print("\nLoading NonverbalTTS dataset from HuggingFace...")
try:
    dataset = load_dataset("deepvk/NonverbalTTS", split="train")
    print(f"[OK] Dataset loaded: {len(dataset)} samples")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Create mapping from metadata to dataset indices
print("\nCreating audio file mapping...")
audio_data = []
skipped = 0

# Process samples in batches to avoid memory issues
batch_size = 100
total_batches = (len(metadata) + batch_size - 1) // batch_size

for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, len(metadata))
    batch_metadata = metadata[start_idx:end_idx]
    
    for meta_item in batch_metadata:
        try:
            # Find corresponding item in dataset
            dataset_idx = meta_item['id']
            if dataset_idx >= len(dataset):
                skipped += 1
                continue
            
            item = dataset[dataset_idx]
            
            # Extract audio
            if 'audio' in item and item['audio'] is not None:
                audio_array = item['audio']['array']
                sample_rate = item['audio']['sampling_rate']
                
                # Save audio file
                audio_filename = f"audio_{dataset_idx:06d}.wav"
                audio_path = output_dir / audio_filename
                
                sf.write(str(audio_path), audio_array, sample_rate)
                
                # Update metadata with audio path
                audio_data.append({
                    'id': dataset_idx,
                    'text': meta_item['text'],
                    'audio_path': str(audio_path),
                    'sample_rate': sample_rate,
                    'duration': meta_item.get('duration', len(audio_array) / sample_rate),
                    'emotion': meta_item.get('emotion', 'neutral'),
                    'speaker_id': meta_item.get('speaker_id', 'unknown'),
                })
            else:
                skipped += 1
                
        except Exception as e:
            print(f"\nWarning: Error processing sample {meta_item.get('id', 'unknown')}: {e}")
            skipped += 1
            continue

print(f"\n[OK] Downloaded {len(audio_data)} audio files")
print(f"     Skipped {skipped} samples")

# Save updated metadata with audio paths
output_metadata = output_dir / "dataset_with_audio.json"
with open(output_metadata, 'w', encoding='utf-8') as f:
    json.dump(audio_data, f, indent=2, ensure_ascii=False)

print(f"\n[OK] Saved metadata with audio paths to: {output_metadata}")
print(f"\nDataset ready for TTS fine-tuning!")
print(f"Total samples: {len(audio_data)}")
print(f"Audio directory: {output_dir}")

