"""
Wait for training to complete (50 epochs) then automatically run test.
"""
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime

checkpoint_dir = Path("checkpoints/dia-nonverbal-lora-gpu")
max_epochs = 50
check_interval = 300  # Check every 5 minutes

print("=" * 70)
print("Automated Training Monitor and Test Runner")
print("=" * 70)
print(f"Monitoring training until {max_epochs} epochs complete...")
print(f"Will check every {check_interval//60} minutes")
print("=" * 70)

while True:
    # Check if training is complete
    if checkpoint_dir.exists():
        epoch_dirs = [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("epoch_")]
        if epoch_dirs:
            epoch_dirs.sort(key=lambda x: int(x.name.split("_")[1]) if x.name.split("_")[1].isdigit() else 0)
            latest_epoch_dir = epoch_dirs[-1]
            latest_epoch = int(latest_epoch_dir.name.split("_")[1])
            
            # Check training history
            hist_file = latest_epoch_dir / "training_history.json"
            if hist_file.exists():
                with open(hist_file) as f:
                    data = json.load(f)
                epochs_completed = len(data.get('epochs', []))
                
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Progress: Epoch {epochs_completed}/{max_epochs}")
                if data.get('losses'):
                    print(f"  Latest loss: {data['losses'][-1]:.6f}")
                    print(f"  Best loss: {min(data['losses']):.6f}")
                
                if epochs_completed >= max_epochs:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Training completed!")
                    break
            else:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Current checkpoint: epoch_{latest_epoch}")
        else:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Training starting...")
    else:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Waiting for training to start...")
    
    time.sleep(check_interval)

# Training complete, run test
print("\n" + "=" * 70)
print("Training Complete! Running test...")
print("=" * 70)

result = subprocess.run(
    ["python", "test_pretrained_model_gpu.py"],
    cwd=Path(__file__).parent
)

if result.returncode == 0:
    print("\n" + "=" * 70)
    print("[SUCCESS] Training and testing completed successfully!")
    print("=" * 70)
else:
    print("\n" + "=" * 70)
    print("[WARNING] Test completed with errors")
    print("=" * 70)

