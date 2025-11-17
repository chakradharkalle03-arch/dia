"""Check GPU availability for PyTorch"""
import torch
import sys

print("=" * 70)
print("GPU Detection Check")
print("=" * 70)

print(f"\nPyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
    print("\n" + "=" * 70)
    print("[OK] GPU is available and ready for training!")
    print("=" * 70)
    sys.exit(0)
else:
    print("\n" + "=" * 70)
    print("[ERROR] CUDA is not available in PyTorch!")
    print("=" * 70)
    print("\nPossible solutions:")
    print("1. Install PyTorch with CUDA support:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("\n2. Check CUDA installation:")
    print("   nvidia-smi")
    sys.exit(1)

