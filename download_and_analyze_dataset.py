"""
Download and analyze NonverbalTTS dataset structure.
This script downloads the dataset and provides detailed analysis.
"""
import os
from datasets import load_dataset
from pathlib import Path
import json
import pandas as pd

# Set HF token from environment variable
if not os.getenv("HF_TOKEN") and not os.getenv("HUGGINGFACE_TOKEN"):
    token = input("Please enter your HuggingFace token: ").strip()
    os.environ["HF_TOKEN"] = token

print("=" * 70)
print("Downloading and Analyzing NonverbalTTS Dataset")
print("=" * 70)

# Download dataset
print("\n1. Downloading dataset from HuggingFace...")
try:
    # Load dataset (this will download if not cached)
    dataset = load_dataset("deepvk/NonverbalTTS", split="train")
    print(f"[OK] Dataset loaded: {len(dataset)} samples")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Trying alternative method...")
    from datasets import load_dataset_builder
    builder = load_dataset_builder("deepvk/NonverbalTTS")
    dataset = builder.as_dataset(split="train")
    print(f"[OK] Dataset loaded: {len(dataset)} samples")

# Analyze dataset structure
print("\n2. Analyzing dataset structure...")
print(f"   Total samples: {len(dataset)}")

# Get dataset features without loading audio
print(f"\n3. Dataset columns/features:")
features = dataset.features
for key in features.keys():
    feature_type = str(features[key])
    print(f"   - {key}: {feature_type}")

# Analyze samples using select to avoid audio decoding
print("\n4. Analyzing sample data...")
sample_data = []
try:
    # Use select to get samples without audio decoding
    sample_indices = list(range(min(5, len(dataset))))
    samples = dataset.select(sample_indices)
    
    for i, sample in enumerate(samples):
        sample_info = {
            "index": i,
            "has_text": "Result" in sample and sample["Result"] is not None,
            "has_audio": "audio" in sample and sample["audio"] is not None,
            "emotion": sample.get("Emotion", "N/A"),
            "speaker_id": sample.get("speaker_id", "N/A"),
            "duration": sample.get("duration", 0.0),
        }
        if "Result" in sample:
            text = sample["Result"]
            sample_info["text_length"] = len(text) if text else 0
            sample_info["text_preview"] = text[:100] if text else ""
        sample_data.append(sample_info)
except Exception as e:
    print(f"   Error: {e}")
    print("   Using alternative method...")
    # Access via features
    for i in range(min(5, len(dataset))):
        try:
            # Get text columns only
            row = dataset[i]
            sample_info = {
                "index": i,
                "emotion": row.get("Emotion", "N/A") if isinstance(row, dict) else "N/A",
                "speaker_id": row.get("speaker_id", "N/A") if isinstance(row, dict) else "N/A",
            }
            sample_data.append(sample_info)
        except:
            pass

print("\n   Sample analysis:")
for sample in sample_data:
    print(f"   Sample {sample['index']}:")
    print(f"     - Emotion: {sample['emotion']}")
    print(f"     - Speaker: {sample['speaker_id']}")
    print(f"     - Duration: {sample['duration']:.2f}s")
    print(f"     - Text length: {sample.get('text_length', 0)}")
    if sample.get('text_preview'):
        print(f"     - Text preview: {sample['text_preview']}...")

# Analyze emotions using select
print("\n5. Emotion distribution:")
emotions = {}
try:
    sample_indices = list(range(min(100, len(dataset))))
    samples = dataset.select(sample_indices)
    for sample in samples:
        emotion = sample.get("Emotion", "unknown")
        emotions[emotion] = emotions.get(emotion, 0) + 1
except Exception as e:
    print(f"   Error: {e}")

for emotion, count in sorted(emotions.items(), key=lambda x: -x[1]):
    print(f"   - {emotion}: {count}")

# Analyze nonverbal tokens using select
print("\n6. Analyzing nonverbal tokens in text...")
nonverbal_tokens = {
    "🌬️": 0, "😤": 0, "😷": 0, "👃": 0, "🤣": 0, 
    "🤧": 0, "🗣️": 0, "😖": 0, "🐖": 0
}

try:
    sample_indices = list(range(min(100, len(dataset))))
    samples = dataset.select(sample_indices)
    for sample in samples:
        text = sample.get("Result", "")
        if text:
            for token in nonverbal_tokens.keys():
                nonverbal_tokens[token] += text.count(token)
except Exception as e:
    print(f"   Error: {e}")

print("   Nonverbal token counts (in first 100 samples):")
for token, count in sorted(nonverbal_tokens.items(), key=lambda x: -x[1]):
    if count > 0:
        print(f"   - {token}: {count}")

# Save analysis
analysis_file = Path("dataset_analysis.json")
analysis = {
    "total_samples": len(dataset),
    "features": list(example.keys()) if len(dataset) > 0 else [],
    "sample_data": sample_data,
    "emotions": emotions,
    "nonverbal_tokens": nonverbal_tokens
}

with open(analysis_file, "w", encoding="utf-8") as f:
    json.dump(analysis, f, indent=2, ensure_ascii=False)

print(f"\n[OK] Analysis saved to: {analysis_file}")
print("\n" + "=" * 70)
print("Dataset Analysis Complete!")
print("=" * 70)

