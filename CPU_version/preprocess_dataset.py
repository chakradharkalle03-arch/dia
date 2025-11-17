"""
Preprocess NonverbalTTS dataset for Dia model training.
Converts emoji tokens to text tags compatible with Dia's byte-level encoding.
"""
import re
from datasets import load_dataset
from pathlib import Path
import json

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
    "👂": "(listens)",
}

def normalize_text(text: str) -> str:
    """Convert emoji tokens to text tags and clean up text."""
    if not text or text == "null":
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
    # Use speaker tag if provided, otherwise default to S1
    speaker_tag = f"[{speaker_id}]" if speaker_id and speaker_id.startswith("S") else "[S1]"
    return f"{speaker_tag} {text}"

def preprocess_dataset():
    """Load and preprocess NonverbalTTS dataset."""
    print("Loading NonverbalTTS dataset...")
    try:
        # Try to load dataset with audio disabled
        dataset = load_dataset("deepvk/NonverbalTTS", split="train", trust_remote_code=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying alternative loading method...")
        # Alternative: load as Arrow dataset
        from datasets import load_dataset_builder
        builder = load_dataset_builder("deepvk/NonverbalTTS")
        dataset = builder.as_dataset(split="train")
    
    print(f"Loaded {len(dataset)} samples")
    
    processed_data = []
    
    # Process dataset without loading audio
    try:
        for idx in range(len(dataset)):
            example = dataset[idx]
            
            # Get the result text (final annotation)
            text = example.get("Result", "")
            if not text or text == "null":
                continue
            
            # Normalize text (convert emojis to tags)
            normalized_text = normalize_text(text)
            if not normalized_text:
                continue
            
            # Format for Dia
            speaker_id = example.get("speaker_id", "S1")
            formatted_text = format_for_dia(normalized_text, speaker_id)
            
            # Get audio info (path only, don't load)
            audio_info = example.get("audio", {})
            if isinstance(audio_info, dict):
                audio_path = audio_info.get("path")
                sample_rate = audio_info.get("sampling_rate", 16000)
            else:
                audio_path = None
                sample_rate = 16000
            
            processed_data.append({
                "id": idx,
                "text": formatted_text,
                "original_text": text,
                "audio_path": audio_path,
                "sample_rate": sample_rate,
                "emotion": example.get("Emotion", "neutral"),
                "speaker_id": speaker_id,
                "duration": example.get("duration", 0.0),
            })
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(dataset)} samples")
    except Exception as e:
        print(f"Error processing dataset: {e}")
        print("Continuing with available data...")
        import traceback
        traceback.print_exc()
    
    print(f"Successfully processed {len(processed_data)} samples")
    
    # Save processed data
    output_dir = Path("processed_data")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "processed_dataset.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved processed dataset to {output_file}")
    return processed_data

if __name__ == "__main__":
    preprocess_dataset()

