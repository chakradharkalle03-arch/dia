"""
GPU Version: Prepare NonverbalTTS dataset for Dia model training.
Converts emoji tokens to text tags and formats for training.
(Can use same dataset as CPU version - shared processed_data folder)
"""
import os
import json
import re
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# Set HF token from environment variable
if not os.getenv("HF_TOKEN") and not os.getenv("HUGGINGFACE_TOKEN"):
    token = input("Please enter your HuggingFace token: ").strip()
    os.environ["HF_TOKEN"] = token

# Mapping from NonverbalTTS emojis to Dia-compatible text tags
EMOJI_TO_TAG = {
    "🌬️": "(breaths)",
    "😤": "(grunts)",
    "😷": "(coughs)",
    "👃": "(sniffs)",
    "🤣": "(laughs)",
    "🤧": "(sneezes)",
    "🗣️": "(vocal_noise)",
    "😖": "(groans)",
    "🐖": "(snorts)",
}

def normalize_text(text: str) -> str:
    """Convert emoji tokens to text tags and clean up text."""
    if not text or text == "null" or text is None:
        return ""
    
    # Replace emojis with text tags
    for emoji, tag in EMOJI_TO_TAG.items():
        text = text.replace(emoji, f" {tag} ")
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def format_for_dia(text: str, speaker_id: str = None) -> str:
    """Format text in Dia's expected format with speaker tags."""
    # Extract speaker from speaker_id if it's in format like "ex01" or "id03621"
    if speaker_id:
        # Use S1 for Expresso speakers (ex01, ex02, etc.), S2 for others
        if speaker_id.startswith("ex"):
            speaker_tag = "[S1]"
        elif speaker_id.startswith("id"):
            speaker_tag = "[S2]"
        else:
            speaker_tag = "[S1]"
    else:
        speaker_tag = "[S1]"
    
    return f"{speaker_tag} {text}"

def prepare_dataset():
    """Load and prepare NonverbalTTS dataset for training."""
    print("=" * 70)
    print("Preparing NonverbalTTS Dataset for Training - GPU VERSION")
    print("=" * 70)
    
    # Load dataset
    print("\n1. Loading dataset...")
    try:
        dataset = load_dataset("deepvk/NonverbalTTS", split="train")
        print(f"[OK] Loaded {len(dataset)} samples")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Process dataset using Arrow format to avoid audio decoding
    print("\n2. Processing dataset...")
    processed_data = []
    skipped = 0
    
    # Convert to Arrow format to access data without decoding audio
    try:
        dataset_arrow = dataset.with_format("arrow")
        table = dataset_arrow[:]
        
        # Extract columns
        result_texts = table["Result"].to_pylist()
        emotions = table["Emotion"].to_pylist()
        speaker_ids = table["speaker_id"].to_pylist()
        durations = table["duration"].to_pylist()
        
        print(f"   Extracted {len(result_texts)} text samples")
        
        # Process each sample
        for idx in tqdm(range(len(result_texts)), desc="Processing"):
            try:
                text = result_texts[idx]
                if not text or text == "null" or text is None:
                    skipped += 1
                    continue
                
                # Normalize text (convert emojis to tags)
                normalized_text = normalize_text(text)
                if not normalized_text:
                    skipped += 1
                    continue
                
                # Format for Dia
                speaker_id = speaker_ids[idx] if idx < len(speaker_ids) else "S1"
                formatted_text = format_for_dia(normalized_text, speaker_id)
                
                processed_data.append({
                    "id": idx,
                    "text": formatted_text,
                    "original_text": text,
                    "normalized_text": normalized_text,
                    "audio_path": None,  # Audio not needed for text-only training
                    "sample_rate": 16000,
                    "emotion": emotions[idx] if idx < len(emotions) else "neutral",
                    "speaker_id": speaker_id,
                    "duration": durations[idx] if idx < len(durations) else 0.0,
                })
            except Exception as e:
                skipped += 1
                continue
                
    except Exception as e:
        print(f"   Error using Arrow format: {e}")
        print("   Falling back to batch processing...")
        # Fallback: process in smaller batches
        batch_size = 50
        for batch_start in tqdm(range(0, len(dataset), batch_size), desc="Processing batches"):
            batch_end = min(batch_start + batch_size, len(dataset))
            batch_indices = list(range(batch_start, batch_end))
            
            try:
                # Select batch and convert to dict format without audio
                batch = dataset.select(batch_indices)
                batch_dict = batch.to_dict()
                
                for local_idx in range(len(batch_dict.get("Result", []))):
                    idx = batch_start + local_idx
                    text = batch_dict["Result"][local_idx]
                    
                    if not text or text == "null":
                        skipped += 1
                        continue
                    
                    normalized_text = normalize_text(text)
                    if not normalized_text:
                        skipped += 1
                        continue
                    
                    speaker_id = batch_dict.get("speaker_id", ["S1"] * len(batch_dict["Result"]))[local_idx]
                    formatted_text = format_for_dia(normalized_text, speaker_id)
                    
                    processed_data.append({
                        "id": idx,
                        "text": formatted_text,
                        "original_text": text,
                        "normalized_text": normalized_text,
                        "audio_path": None,
                        "sample_rate": 16000,
                        "emotion": batch_dict.get("Emotion", ["neutral"] * len(batch_dict["Result"]))[local_idx],
                        "speaker_id": speaker_id,
                        "duration": batch_dict.get("duration", [0.0] * len(batch_dict["Result"]))[local_idx],
                    })
            except Exception as e2:
                skipped += batch_end - batch_start
                continue
    
    print(f"\n[OK] Processed {len(processed_data)} samples")
    print(f"  Skipped {skipped} samples")
    
    # Save processed data (use parent directory to share with CPU version)
    output_dir = Path("../processed_data")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "processed_dataset.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Saved processed dataset to: {output_file}")
    
    # Show statistics
    print("\n3. Dataset Statistics:")
    emotions = {}
    for item in processed_data:
        emotion = item.get("emotion", "unknown")
        emotions[emotion] = emotions.get(emotion, 0) + 1
    
    print(f"   Total samples: {len(processed_data)}")
    print(f"   Emotions:")
    for emotion, count in sorted(emotions.items(), key=lambda x: -x[1]):
        print(f"     - {emotion}: {count}")
    
    # Show sample
    if len(processed_data) > 0:
        print("\n4. Sample processed data:")
        sample = processed_data[0]
        try:
            print(f"   Original: {sample['original_text'][:80]}...")
        except:
            print(f"   Original: [contains emoji]")
        print(f"   Formatted: {sample['text'][:80]}...")
        print(f"   Emotion: {sample['emotion']}")
        print(f"   Speaker: {sample['speaker_id']}")
    
    print("\n" + "=" * 70)
    print("Dataset Preparation Complete!")
    print("=" * 70)
    
    return processed_data

if __name__ == "__main__":
    prepare_dataset()

