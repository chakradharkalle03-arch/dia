"""
Hugging Face Token Setup Script
Helps users set up their HF_TOKEN environment variable.
"""
import os
import sys

def setup_hf_token():
    """Interactive script to set up Hugging Face token."""
    print("=" * 70)
    print("Hugging Face Token Setup")
    print("=" * 70)
    print("\nThis script will help you set up your Hugging Face API token.")
    print("You need this token to download the Dia model and dataset.")
    print("\nIf you don't have a token yet:")
    print("  1. Go to: https://huggingface.co/join")
    print("  2. Create a free account")
    print("  3. Go to: https://huggingface.co/settings/tokens")
    print("  4. Create a new token with 'Read' permission")
    print("  5. Copy the token (it starts with 'hf_')")
    print("=" * 70)
    
    # Check if token already exists
    existing_token = os.environ.get("HF_TOKEN")
    if existing_token:
        print(f"\n[INFO] Found existing token: {existing_token[:10]}...")
        use_existing = input("Use existing token? (y/n): ").strip().lower()
        if use_existing == 'y':
            print("\n[OK] Using existing token")
            return existing_token
    
    # Get token from user
    print("\nEnter your Hugging Face token:")
    print("(It should start with 'hf_' and be about 40 characters long)")
    token = input("Token: ").strip()
    
    if not token:
        print("\n[ERROR] Token cannot be empty!")
        sys.exit(1)
    
    if not token.startswith("hf_"):
        print("\n[WARNING] Token should start with 'hf_'")
        continue_anyway = input("Continue anyway? (y/n): ").strip().lower()
        if continue_anyway != 'y':
            print("\n[INFO] Setup cancelled")
            sys.exit(0)
    
    # Set token for current session
    os.environ["HF_TOKEN"] = token
    print("\n[OK] Token set for current session")
    
    # Instructions for permanent setup
    print("\n" + "=" * 70)
    print("To make this permanent:")
    print("=" * 70)
    
    if sys.platform == "win32":
        print("\nWindows - Temporary (current session only):")
        print(f'  set HF_TOKEN="{token}"')
        print("\nWindows - Permanent (recommended):")
        print(f'  setx HF_TOKEN "{token}"')
        print("  (Then restart Command Prompt)")
    else:
        print("\nMac/Linux - Temporary (current session only):")
        print(f'  export HF_TOKEN="{token}"')
        print("\nMac/Linux - Permanent (recommended):")
        print(f'  echo \'export HF_TOKEN="{token}"\' >> ~/.bashrc')
        print("  source ~/.bashrc")
        print("\nOr add to ~/.zshrc if using zsh:")
        print(f'  echo \'export HF_TOKEN="{token}"\' >> ~/.zshrc')
        print("  source ~/.zshrc")
    
    print("\n" + "=" * 70)
    print("Verification:")
    print("=" * 70)
    print("\nTest if token is set:")
    print('  python -c "import os; print(\'Token set:\', \\"HF_TOKEN\\" in os.environ)"')
    
    print("\n[SUCCESS] Token setup complete!")
    print("\nYou can now:")
    print("  1. Run: python prepare_training_data.py (to download dataset)")
    print("  2. Run: python test_pretrained_model.py (to download model and test)")
    
    return token

if __name__ == "__main__":
    try:
        setup_hf_token()
    except KeyboardInterrupt:
        print("\n\n[INFO] Setup cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        sys.exit(1)

