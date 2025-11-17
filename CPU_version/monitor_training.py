"""Monitor training progress"""
import time
import json
from pathlib import Path
from datetime import datetime

checkpoint_dir = Path("checkpoints/dia-nonverbal-lora")
log_file = Path("training_log.txt")

print("=" * 70)
print("Training Monitor - Checking Progress")
print("=" * 70)

# Check log file
if log_file.exists():
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        if lines:
            print(f"\nLast {min(20, len(lines))} lines from training log:")
            for line in lines[-20:]:
                print(f"  {line.rstrip()}")
        else:
            print("\nLog file is empty")
else:
    print("\nNo log file found yet")

# Check checkpoints
if checkpoint_dir.exists():
    epoch_dirs = [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("epoch_")]
    epoch_dirs.sort(key=lambda x: int(x.name.split("_")[1]) if x.name.split("_")[1].isdigit() else 0)
    
    if epoch_dirs:
        latest = epoch_dirs[-1]
        print(f"\nLatest checkpoint: {latest.name}")
        print(f"  Modified: {datetime.fromtimestamp(latest.stat().st_mtime)}")
        
        # Check for training history
        hist_file = latest / "training_history.json"
        if hist_file.exists():
            with open(hist_file) as f:
                data = json.load(f)
            print(f"  Epochs completed: {len(data.get('epochs', []))}")
            if data.get('losses'):
                print(f"  Latest loss: {data['losses'][-1]:.6f}")
                print(f"  Best loss: {min(data['losses']):.6f}")
        else:
            print("  No training history yet")
        
        print(f"\nTotal checkpoints: {len(epoch_dirs)}")
    else:
        print("\nNo checkpoints found yet")
else:
    print("\nCheckpoint directory doesn't exist yet")

print("\n" + "=" * 70)
