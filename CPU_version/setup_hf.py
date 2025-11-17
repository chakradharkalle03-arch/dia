"""
Setup HuggingFace API key and download NonverbalTTS dataset.
"""
import os
from huggingface_hub import login, snapshot_download
from datasets import load_dataset

# Set API key from environment variable or user input
api_key = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
if not api_key:
    api_key = input("Please enter your HuggingFace token: ").strip()

print("Logging into HuggingFace...")
try:
    login(token=api_key, add_to_git_credential=True)
    print("[OK] Successfully logged into HuggingFace")
except Exception as e:
    print(f"Error logging in: {e}")
    # Set as environment variable instead
    os.environ["HF_TOKEN"] = api_key
    print("[OK] Set HF_TOKEN environment variable")

print("\nDownloading NonverbalTTS dataset...")
try:
    # Load dataset (this will download it)
    dataset = load_dataset("deepvk/NonverbalTTS", split="train")
    print(f"[OK] Dataset downloaded successfully!")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Features: {list(dataset.features.keys())}")
except Exception as e:
    print(f"Error downloading dataset: {e}")
    import traceback
    traceback.print_exc()

print("\nDataset setup complete!")
