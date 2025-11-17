"""
Fix torchvision compatibility issue by reinstalling compatible versions.
"""
import subprocess
import sys

print("=" * 70)
print("Fixing Torchvision Compatibility Issue")
print("=" * 70)

print("\nThe issue is a version mismatch between PyTorch and Torchvision.")
print("Installing compatible versions...\n")

# Uninstall problematic packages
print("Step 1: Uninstalling torchvision...")
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchvision"], check=False)

# Install compatible torchvision
print("\nStep 2: Installing compatible torchvision...")
subprocess.run([sys.executable, "-m", "pip", "install", "torchvision==0.20.0"], check=True)

print("\n" + "=" * 70)
print("Fix Complete!")
print("=" * 70)
print("\nPlease try running the training script again:")
print("  cd GPU_version")
print("  python train_speecht5_lora_gpu_fixed.py")

