"""
Explore and understand the NonverbalTTS dataset structure.
This script helps understand the data format and prepares it for training.
"""
import json
from pathlib import Path
from datasets import load_dataset
import pandas as pd

print("=" * 70)
print("NonverbalTTS Dataset Exploration")
print("=" * 70)

# Load dataset
print("\n1. Loading dataset...")
try:
    dataset = load_dataset("deepvk/NonverbalTTS", split="train")
    print(f"[OK] Loaded {len(dataset)} samples")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Trying alternative method...")
    # Try loading without audio decoding
    try:
        from datasets import load_dataset_builder
        builder = load_dataset_builder("deepvk/NonverbalTTS")
        dataset = builder.as_dataset(split="train")
        print(f"✓ Loaded {len(dataset)} samples (without audio)")
    except Exception as e2:
        print(f"Error: {e2}")
        exit(1)

# Explore features
print("\n2. Dataset Features:")
print(f"   Features: {list(dataset.features.keys())}")

# Sample a few examples (access raw data without audio decoding)
print("\n3. Sample Data Analysis:")
print("-" * 70)

sample_count = min(5, len(dataset))
for i in range(sample_count):
    try:
        # Access raw data from Arrow table to avoid audio decoding
        example = dataset[i]
        print(f"\nSample {i+1}:")
        print(f"  Keys: {list(example.keys())}")
        
        # Text fields (these should work)
        if 'Result' in example:
            result_text = str(example['Result'])
            print(f"  Result text: {result_text[:100]}..." if len(result_text) > 100 else f"  Result text: {result_text}")
        
        if 'Initial text' in example:
            initial_text = str(example['Initial text'])
            print(f"  Initial text: {initial_text[:100]}..." if len(initial_text) > 100 else f"  Initial text: {initial_text}")
        
        # Metadata
        if 'Emotion' in example:
            print(f"  Emotion: {example['Emotion']}")
        if 'speaker_id' in example:
            print(f"  Speaker ID: {example['speaker_id']}")
        if 'duration' in example:
            print(f"  Duration: {example['duration']}s")
        if 'dnsmos' in example:
            print(f"  DNSMOS: {example['dnsmos']}")
        
        # Audio info (just path, don't load)
        if 'audio' in example:
            try:
                audio_info = example['audio']
                if isinstance(audio_info, dict):
                    print(f"  Audio path: {audio_info.get('path', 'N/A')}")
                    print(f"  Sample rate: {audio_info.get('sampling_rate', 'N/A')}")
            except:
                print(f"  Audio: (cannot decode - FFmpeg issue)")
        
        # Annotator responses
        if 'Annotator response 1' in example:
            ann1 = str(example['Annotator response 1'])
            print(f"  Annotator 1: {ann1[:80]}...")
        
    except Exception as e:
        print(f"  Error reading sample {i}: {e}")
        # Try accessing via Arrow table directly
        try:
            table = dataset._data
            row = table[i]
            print(f"  Raw data available via Arrow table")
        except:
            continue

# Statistics
print("\n4. Dataset Statistics:")
print("-" * 70)

try:
    # Convert to pandas for easier analysis - use Arrow table directly
    df_data = []
    table = dataset._data
    for i in range(min(100, len(dataset))):  # Sample first 100 for stats
        try:
            row = table[i]
            df_data.append({
                'emotion': row['Emotion'].as_py() if hasattr(row['Emotion'], 'as_py') else row['Emotion'],
                'speaker_id': row['speaker_id'].as_py() if hasattr(row['speaker_id'], 'as_py') else row['speaker_id'],
                'duration': row['duration'].as_py() if hasattr(row['duration'], 'as_py') else row['duration'],
                'dnsmos': row['dnsmos'].as_py() if hasattr(row['dnsmos'], 'as_py') else row['dnsmos'],
                'has_result': row['Result'] is not None and str(row['Result']) != 'null',
            })
        except Exception as e:
            # Fallback to regular access
            try:
                example = dataset[i]
                df_data.append({
                    'emotion': example.get('Emotion', 'unknown'),
                    'speaker_id': example.get('speaker_id', 'unknown'),
                    'duration': example.get('duration', 0),
                    'dnsmos': example.get('dnsmos', 0),
                    'has_result': example.get('Result') is not None and str(example.get('Result')) != 'null',
                })
            except:
                continue
    
    if df_data:
        df = pd.DataFrame(df_data)
        
        print(f"\nEmotion distribution:")
        print(df['emotion'].value_counts())
        
        print(f"\nDuration statistics:")
        print(f"  Mean: {df['duration'].mean():.2f}s")
        print(f"  Min: {df['duration'].min():.2f}s")
        print(f"  Max: {df['duration'].max():.2f}s")
        
        print(f"\nDNSMOS (quality) statistics:")
        print(f"  Mean: {df['dnsmos'].mean():.2f}")
        print(f"  Min: {df['dnsmos'].min():.2f}")
        print(f"  Max: {df['dnsmos'].max():.2f}")
        
        print(f"\nUnique speakers: {df['speaker_id'].nunique()}")
        print(f"Samples with valid Result: {df['has_result'].sum()}/{len(df)}")
        
except Exception as e:
    print(f"Error computing statistics: {e}")

# Check for nonverbal tokens
print("\n5. Nonverbal Token Analysis:")
print("-" * 70)

emoji_tokens = {
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

token_counts = {token: 0 for token in emoji_tokens.keys()}

# Access via Arrow table to avoid audio decoding
table = dataset._data
for i in range(min(100, len(dataset))):
    try:
        row = table[i]
        result_text = str(row['Result'])
        for emoji in emoji_tokens.keys():
            if emoji in result_text:
                token_counts[emoji] += 1
    except:
        # Fallback
        try:
            example = dataset[i]
            result_text = str(example.get('Result', ''))
            for emoji in emoji_tokens.keys():
                if emoji in result_text:
                    token_counts[emoji] += 1
        except:
            continue

print("\nNonverbal token frequency (in first 100 samples):")
for emoji, count in sorted(token_counts.items(), key=lambda x: x[1], reverse=True):
    if count > 0:
        print(f"  {emoji} ({emoji_tokens[emoji]}): {count} occurrences")

# Save exploration results
print("\n6. Saving exploration results...")
output_file = Path("dataset_exploration.json")
exploration_data = {
    "total_samples": len(dataset),
    "features": list(dataset.features.keys()),
    "token_counts": token_counts,
    "sample_examples": []
}

for i in range(min(3, len(dataset))):
    try:
        example = dataset[i]
        exploration_data["sample_examples"].append({
            "id": i,
            "emotion": example.get('Emotion'),
            "speaker_id": example.get('speaker_id'),
            "result_text": str(example.get('Result', ''))[:200],
            "duration": example.get('duration'),
        })
    except:
        continue

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(exploration_data, f, indent=2, ensure_ascii=False)

print(f"[OK] Saved exploration results to {output_file}")

print("\n" + "=" * 70)
print("Exploration Complete!")
print("=" * 70)
