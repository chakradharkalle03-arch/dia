"""Check training progress"""
import json
from pathlib import Path

checkpoint_dir = Path("checkpoints/dia-nonverbal-lora")
if not checkpoint_dir.exists():
    print("No checkpoints directory found")
    exit(1)

# Find all epoch directories
epoch_dirs = [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("epoch_")]
epoch_dirs.sort(key=lambda x: int(x.name.split("_")[1]) if x.name.split("_")[1].isdigit() else 0)

if not epoch_dirs:
    print("No checkpoints found")
    exit(1)

latest = epoch_dirs[-1]
print(f"Latest checkpoint: {latest.name}")

# Check for training history
hist_file = latest / "training_history.json"
if hist_file.exists():
    with open(hist_file) as f:
        data = json.load(f)
    print(f"\nTraining Progress:")
    print(f"  Epochs completed: {len(data.get('epochs', []))}")
    if data.get('losses'):
        print(f"  Latest loss: {data['losses'][-1]:.6f}")
        print(f"  Best loss: {min(data['losses']):.6f}")
    if data.get('learning_rates'):
        print(f"  Current LR: {data['learning_rates'][-1]:.2e}")
else:
    print("No training history file found")

# Check checkpoint file
ckpt_file = latest / "checkpoint.pt"
if ckpt_file.exists():
    import torch
    ckpt = torch.load(ckpt_file, map_location='cpu')
    print(f"\nCheckpoint Info:")
    print(f"  Epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"  Loss: {ckpt.get('loss', 'N/A')}")
    print(f"  Best Loss: {ckpt.get('best_loss', 'N/A')}")

print(f"\nTotal checkpoints: {len(epoch_dirs)}")

