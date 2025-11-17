"""
Automated script to train and test the model on GPU.
Monitors training progress and automatically runs test after completion.
"""
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime

def check_training_complete(checkpoint_dir: Path, max_epochs: int = 50):
    """Check if training has completed all epochs."""
    if not checkpoint_dir.exists():
        return False
    
    epoch_dirs = [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("epoch_")]
    if not epoch_dirs:
        return False
    
    epoch_dirs.sort(key=lambda x: int(x.name.split("_")[1]) if x.name.split("_")[1].isdigit() else 0)
    latest_epoch = int(epoch_dirs[-1].name.split("_")[1])
    
    # Check training history
    hist_file = epoch_dirs[-1] / "training_history.json"
    if hist_file.exists():
        with open(hist_file) as f:
            data = json.load(f)
        epochs_completed = len(data.get('epochs', []))
        return epochs_completed >= max_epochs
    
    return latest_epoch >= max_epochs

def monitor_training(checkpoint_dir: Path, max_epochs: int = 50, check_interval: int = 300):
    """Monitor training progress."""
    print("=" * 70)
    print("Monitoring GPU Training Progress")
    print("=" * 70)
    print(f"Will check every {check_interval//60} minutes...")
    print(f"Target: {max_epochs} epochs")
    print("=" * 70)
    
    while True:
        if check_training_complete(checkpoint_dir, max_epochs):
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Training completed!")
            return True
        
        # Check current progress
        if checkpoint_dir.exists():
            epoch_dirs = [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("epoch_")]
            if epoch_dirs:
                epoch_dirs.sort(key=lambda x: int(x.name.split("_")[1]) if x.name.split("_")[1].isdigit() else 0)
                latest_epoch = int(epoch_dirs[-1].name.split("_")[1])
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Current progress: Epoch {latest_epoch}/{max_epochs}")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Training starting...")
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for training to start...")
        
        time.sleep(check_interval)

def main():
    """Main function to train and test."""
    checkpoint_dir = Path("checkpoints/dia-nonverbal-lora-gpu")
    max_epochs = 50
    
    print("=" * 70)
    print("Automated GPU Training and Testing")
    print("=" * 70)
    
    # Step 1: Start training
    print("\n[STEP 1] Starting GPU training...")
    print("Training script: train_dia_proper_gpu.py")
    print("This will take several hours for 50 epochs...")
    
    # Training is already running in background, just monitor
    print("\n[STEP 2] Monitoring training progress...")
    monitor_training(checkpoint_dir, max_epochs)
    
    # Step 3: Run test
    print("\n" + "=" * 70)
    print("[STEP 3] Training Complete! Starting test...")
    print("=" * 70)
    
    print("\nRunning test script: test_pretrained_model_gpu.py")
    result = subprocess.run(
        ["python", "test_pretrained_model_gpu.py"],
        cwd=Path(__file__).parent,
        capture_output=False
    )
    
    if result.returncode == 0:
        print("\n" + "=" * 70)
        print("[SUCCESS] Training and testing completed successfully!")
        print("=" * 70)
        print(f"\nCheckpoints saved in: {checkpoint_dir}")
        print("Generated audio saved in: generated_audio/")
    else:
        print("\n" + "=" * 70)
        print("[WARNING] Test completed with errors")
        print("=" * 70)

if __name__ == "__main__":
    main()

