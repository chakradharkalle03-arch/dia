"""
Live monitoring script for training progress.
Run this to see real-time training status.
"""
import json
import time
from pathlib import Path
import subprocess
import sys

def check_training_status():
    """Check current training status."""
    print("=" * 70)
    print("Training Status Monitor")
    print("=" * 70)
    
    # Check if process is running
    try:
        result = subprocess.run(
            ["powershell", "-Command", "Get-Process python -ErrorAction SilentlyContinue | Measure-Object | Select-Object -ExpandProperty Count"],
            capture_output=True,
            text=True
        )
        process_count = int(result.stdout.strip())
        print(f"\n✅ Python processes running: {process_count}")
    except:
        print("\n⚠️  Could not check process status")
    
    # Check CPU training
    cpu_checkpoint = Path("CPU_version/checkpoints/speecht5_lora_cpu")
    if cpu_checkpoint.exists():
        print(f"\n📁 CPU Checkpoints directory: {cpu_checkpoint}")
        
        # Find latest checkpoint
        epochs = [d for d in cpu_checkpoint.iterdir() if d.is_dir() and d.name.startswith("epoch_")]
        if epochs:
            latest = max(epochs, key=lambda x: int(x.name.split("_")[1]))
            print(f"   Latest checkpoint: {latest.name}")
            
            # Check training history
            history_file = latest / "training_history.json"
            if not history_file.exists():
                history_file = cpu_checkpoint / "training_history.json"
            
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history = json.load(f)
                    if isinstance(history, list) and len(history) > 0:
                        latest_epoch = history[-1]
                        print(f"   Latest epoch: {latest_epoch.get('epoch', 'N/A')}")
                        print(f"   Latest loss: {latest_epoch.get('loss', 'N/A'):.6f}")
                        print(f"   Learning rate: {latest_epoch.get('lr', 'N/A'):.2e}")
    
    # Check GPU training
    gpu_checkpoint = Path("GPU_version/checkpoints/speecht5_lora_gpu")
    if gpu_checkpoint.exists():
        print(f"\n📁 GPU Checkpoints directory: {gpu_checkpoint}")
        epochs = [d for d in gpu_checkpoint.iterdir() if d.is_dir() and d.name.startswith("epoch_")]
        if epochs:
            latest = max(epochs, key=lambda x: int(x.name.split("_")[1]))
            print(f"   Latest checkpoint: {latest.name}")
    else:
        print("\n📁 GPU Checkpoints: Not started yet")
    
    print("\n" + "=" * 70)
    print("Training is running in the background.")
    print("Checkpoints are saved automatically every 5 epochs.")
    print("=" * 70)

if __name__ == "__main__":
    check_training_status()
    print("\n💡 Tip: Run this script again to see updated status")

