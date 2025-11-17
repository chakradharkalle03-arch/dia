"""
Start fine-tuning for both CPU and GPU versions.
This script will automatically detect available resources and start training.
"""
import os
import sys
import subprocess
import torch
from pathlib import Path

def check_requirements():
    """Check if all requirements are met."""
    print("=" * 70)
    print("Checking Requirements for Fine-tuning")
    print("=" * 70)
    
    # Check if audio dataset exists
    audio_dataset = Path("dataset_audio/dataset_with_audio.json")
    if not audio_dataset.exists():
        print("\n[WARNING] Audio dataset not found!")
        print("Please run: python download_audio_dataset.py")
        print("This may take a while (downloading 17 hours of audio)...")
        return False
    
    print(f"\n[OK] Audio dataset found: {audio_dataset}")
    
    # Check HuggingFace token
    if not os.getenv("HF_TOKEN") and not os.getenv("HUGGINGFACE_TOKEN"):
        print("\n[WARNING] HuggingFace token not set!")
        print("Set it with: set HF_TOKEN=your_token (Windows) or export HF_TOKEN=your_token (Linux/Mac)")
        token = input("\nEnter your HuggingFace token now (or press Enter to skip): ").strip()
        if token:
            os.environ["HF_TOKEN"] = token
            print("[OK] Token set")
        else:
            print("[WARNING] Continuing without token (may fail if model needs authentication)")
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"\n[INFO] CUDA available: {cuda_available}")
    if cuda_available:
        print(f"       GPU: {torch.cuda.get_device_name(0)}")
        print(f"       Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    return True


def start_cpu_training():
    """Start CPU training."""
    print("\n" + "=" * 70)
    print("Starting CPU Training")
    print("=" * 70)
    
    script_path = Path("CPU_version/train_speecht5_cpu.py")
    if not script_path.exists():
        print(f"Error: Training script not found at {script_path}")
        return False
    
    print(f"\nRunning: python {script_path}")
    print("Note: CPU training will be slower. This may take several hours.")
    print("\nYou can monitor progress in the output above.")
    print("Press Ctrl+C to stop training.\n")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=Path.cwd(),
            check=False
        )
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return False
    except Exception as e:
        print(f"\nError during CPU training: {e}")
        return False


def start_gpu_training():
    """Start GPU training."""
    if not torch.cuda.is_available():
        print("\n[SKIP] GPU training skipped - CUDA not available")
        return False
    
    print("\n" + "=" * 70)
    print("Starting GPU Training")
    print("=" * 70)
    
    script_path = Path("GPU_version/train_speecht5_gpu.py")
    if not script_path.exists():
        print(f"Error: Training script not found at {script_path}")
        return False
    
    print(f"\nRunning: python {script_path}")
    print("GPU training will be faster than CPU.")
    print("\nYou can monitor progress in the output above.")
    print("Press Ctrl+C to stop training.\n")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=Path.cwd(),
            check=False
        )
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return False
    except Exception as e:
        print(f"\nError during GPU training: {e}")
        return False


def main():
    """Main function to start both CPU and GPU training."""
    print("=" * 70)
    print("TTS Fine-tuning Starter - CPU and GPU Versions")
    print("=" * 70)
    
    # Check requirements
    if not check_requirements():
        print("\n[ERROR] Requirements not met. Please fix the issues above.")
        return
    
    # Ask user which to run
    print("\n" + "=" * 70)
    print("Training Options")
    print("=" * 70)
    print("\n1. CPU version only")
    print("2. GPU version only (if CUDA available)")
    print("3. Both CPU and GPU (sequential)")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        start_cpu_training()
    elif choice == "2":
        start_gpu_training()
    elif choice == "3":
        print("\n[INFO] Starting CPU training first, then GPU...")
        cpu_success = start_cpu_training()
        if cpu_success:
            print("\n[INFO] CPU training completed. Starting GPU training...")
            start_gpu_training()
        else:
            print("\n[WARNING] CPU training had issues. Skipping GPU training.")
    elif choice == "4":
        print("\nExiting...")
        return
    else:
        print("\nInvalid choice. Exiting...")
        return
    
    print("\n" + "=" * 70)
    print("Fine-tuning Process Complete!")
    print("=" * 70)
    print("\nCheckpoints saved in:")
    print("  - CPU: CPU_version/checkpoints/speecht5_finetuned_cpu/")
    print("  - GPU: GPU_version/checkpoints/speecht5_finetuned_gpu/")
    print("\nTo test your models, run:")
    print("  python test_finetuned_model.py --checkpoint <checkpoint_path>")


if __name__ == "__main__":
    main()

