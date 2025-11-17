"""
Create a sample dataset for testing training setup.
This creates a minimal dataset with a few examples.
"""
import json
from pathlib import Path

# Sample data based on NonverbalTTS format
sample_data = [
    {
        "id": 0,
        "text": "[S1] Hey, hey, stop. Let's not get into what we look like (laughs) at 10.",
        "original_text": "Hey, hey, stop. Let's not get into what we look like 🤣 at 10.",
        "audio_path": None,
        "sample_rate": 16000,
        "emotion": "happy",
        "speaker_id": "S1",
        "duration": 4.82
    },
    {
        "id": 1,
        "text": "[S1] I mean, yeah, it makes me a little bit scared, but it also makes me a little bit relaxed in the way that, like, (breaths) if the ocean is so vast and we're part of all of that, we are a part of this.",
        "original_text": "I mean, yeah, it makes me a little bit scared, but it also makes me a little bit relaxed in the way that, like, 🌬️ if the ocean is so vast and we're part of all of that, we are a part of this.",
        "audio_path": None,
        "sample_rate": 16000,
        "emotion": "other",
        "speaker_id": "S1",
        "duration": 15.0
    },
    {
        "id": 2,
        "text": "[S1] The entire movie when I was watching it, I was kind of chuckling a little bit (breaths) because for me, I was like, oh, I remember that. It was really funny. We were having some jokes afterwards and went out to eat (breaths) sandwiches after that.",
        "original_text": "The entire movie when I was watching it, I was kind of chuckling a little bit 🌬️ because for me, I was like, oh, I remember that. It was really funny. We were having some jokes afterwards and went out to eat 🌬️ sandwiches after that.",
        "audio_path": None,
        "sample_rate": 16000,
        "emotion": "happy",
        "speaker_id": "S1",
        "duration": 12.544
    },
    {
        "id": 3,
        "text": "[S1] I neededed to. I didn't really know, I guess, how to or I didn't really know what to say, but anyways, I know I'm going to Chicago and I'm having a talk at UIC, so (breaths) I know that a lot of questions will be asked.",
        "original_text": "I neededed to. I didn't really know, I guess, how to or I didn't really know what to say, but anyways, I know I'm going to Chicago and I'm having a talk at UIC, so 🌬️ I know that a lot of questions will be asked.",
        "audio_path": None,
        "sample_rate": 16000,
        "emotion": "neutral",
        "speaker_id": "S1",
        "duration": 12.32
    },
    {
        "id": 4,
        "text": "[S1] watch the sunrise with me and we'd breathe in the (breaths) ocean air and",
        "original_text": "watch the sunrise with me and we'd breathe in the 🌬️ ocean air and",
        "audio_path": None,
        "sample_rate": 16000,
        "emotion": "sad",
        "speaker_id": "S1",
        "duration": 7.64
    },
    {
        "id": 5,
        "text": "[S1] It was a good time. (breaths) (grunts)",
        "original_text": "It was a good time. 🌬️ 😤",
        "audio_path": None,
        "sample_rate": 16000,
        "emotion": "happy",
        "speaker_id": "S1",
        "duration": 3.08
    },
    {
        "id": 6,
        "text": "[S1] Mark is a father from California, Jenny Beth (breaths) is a mother from Atlanta. (breaths) And I went to their convention just to hang out. I mean, I wasn't speaking, but just to hang out to meet these people and talk to them. (breaths) And I saw two thousand people.",
        "original_text": "Mark is a father from California, Jenny Beth 🌬️ is a mother from Atlanta. 🌬️ And I went to their convention just to hang out. I mean, I wasn't speaking, but just to hang out to meet these people and talk to them. 🌬️ And I saw two thousand people.",
        "audio_path": None,
        "sample_rate": 16000,
        "emotion": "neutral",
        "speaker_id": "S1",
        "duration": 14.976
    },
    {
        "id": 7,
        "text": "[S1] It's no different really and I (coughs) also met with a guy named Ethan who was a real, you know, who was a guy who went blind at nineteen and his girlfriend.",
        "original_text": "It's no different really and I 😷 also met with a guy named Ethan who was a real, you know, who was a guy who went blind at nineteen and his girlfriend.",
        "audio_path": None,
        "sample_rate": 16000,
        "emotion": "neutral",
        "speaker_id": "S1",
        "duration": 7.424
    },
    {
        "id": 8,
        "text": "[S1] So, Mom, (breaths) how've you been?",
        "original_text": "So, Mom, 🌬️ how've you been?",
        "audio_path": None,
        "sample_rate": 16000,
        "emotion": "sad",
        "speaker_id": "S1",
        "duration": 3.63
    },
    {
        "id": 9,
        "text": "[S1] Yes, yes, you're right. I messed up my words there. Thank you. (laughs) Yep, thank you. (laughs) Good catch. So Hollywood has this thing where they keep making books into movies and they need to",
        "original_text": "Yes, yes, you're right. I messed up my words there. Thank you. 🤣 Yep, thank you. 🤣 Good catch. So Hollywood has this thing where they keep making books into movies and they need to",
        "audio_path": None,
        "sample_rate": 16000,
        "emotion": "happy",
        "speaker_id": "S1",
        "duration": 15.0
    }
]

# Create output directory
output_dir = Path("processed_data")
output_dir.mkdir(exist_ok=True)

# Save sample dataset
output_file = output_dir / "processed_dataset.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(sample_data, f, indent=2, ensure_ascii=False)

print(f"Created sample dataset with {len(sample_data)} examples")
print(f"Saved to: {output_file}")
print("\nSample entries:")
for i, item in enumerate(sample_data[:3]):
    print(f"\n{i+1}. {item['text'][:80]}...")
    print(f"   Emotion: {item['emotion']}, Duration: {item['duration']}s")

