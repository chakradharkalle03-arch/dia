"""Ensure training completes all 50 epochs"""
import time
import json
import subprocess
from pathlib import Path

checkpoint_dir = Path("checkpoints/dia-nonverbal-lora")
max_epochs = 50
check_interval = 300  # Check every 5 minutes

print("=" * 70)
print("Training Completion Monitor")
print("=" * 70)
print(f"Will monitor until {max_epochs} epochs are complete")
print(f"Checking every {check_interval//60} minutes...")
print("=" * 70)

while True:
    # Check if training is still running
    try:
        result = subprocess.run(
            ["powershell", "-Command", "Get-Process python -ErrorAction SilentlyContinue | Where-Object {$_.CommandLine -like '*train_dia_proper*'} | Measure-Object | Select-Object -ExpandProperty Count"],
            capture_output=True,
            text=True,
            timeout=5
        )
        is_running = result.stdout.strip() != "0" and result.stdout.strip() != ""
    except:
        is_running = True  # Assume running if we can't check
    
    # Check checkpoints
    if checkpoint_dir.exists():
        epoch_dirs = [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("epoch_")]
        epoch_dirs.sort(key=lambda x: int(x.name.split("_")[1]) if x.name.split("_")[1].isdigit() else 0)
        
        if epoch_dirs:
            latest_epoch = int(epoch_dirs[-1].name.split("_")[1])
            print(f"\n[{time.strftime('%H:%M:%S')}] Current progress: Epoch {latest_epoch}/{max_epochs}")
            
            # Check training history
            hist_file = epoch_dirs[-1] / "training_history.json"
            if hist_file.exists():
                with open(hist_file) as f:
                    data = json.load(f)
                epochs_completed = len(data.get('epochs', []))
                print(f"  Epochs completed: {epochs_completed}")
                if data.get('losses'):
                    print(f"  Latest loss: {data['losses'][-1]:.6f}")
            
            if latest_epoch >= max_epochs:
                print(f"\n[SUCCESS] Training completed! Reached epoch {latest_epoch}")
                break
        else:
            print(f"\n[{time.strftime('%H:%M:%S')}] No checkpoints yet, training starting...")
    else:
        print(f"\n[{time.strftime('%H:%M:%S')}] Checkpoint directory not found yet...")
    
    if not is_running and checkpoint_dir.exists():
        epoch_dirs = [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("epoch_")]
        if epoch_dirs:
            latest_epoch = int(sorted(epoch_dirs, key=lambda x: int(x.name.split("_")[1]) if x.name.split("_")[1].isdigit() else 0)[-1].name.split("_")[1])
            if latest_epoch < max_epochs:
                print(f"\n[WARNING] Training process stopped at epoch {latest_epoch}. Restarting...")
                # Restart training
                subprocess.Popen(["python", "train_dia_proper.py"], cwd=Path.cwd())
                time.sleep(10)
    
    time.sleep(check_interval)

print("\n" + "=" * 70)
print("Training Monitoring Complete!")
print("=" * 70)

