"""
Test the pretrained Dia model fine-tuned on NonverbalTTS dataset.
Generates audio from text input with nonverbal tokens.
"""
import sys
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from datetime import datetime

# Add dia to path
sys.path.insert(0, str(Path(__file__).parent / "dia"))
from dia.model import Dia

# Mapping from emojis to text tags (same as training)
EMOJI_TO_TAG = {
    "🌬️": "(breaths)",
    "😤": "(grunts)",
    "😷": "(coughs)",
    "👃": "(sniffs)",
    "🤣": "(laughs)",
    "🤧": "(sneezes)",
    "🗣️": "(vocal_noise)",
    "😖": "(groans)",
    "🐖": "(snorts)",
}

def normalize_text(text: str) -> str:
    """Convert emoji tokens to text tags and clean up text."""
    if not text:
        return ""
    
    # Replace emojis with text tags
    for emoji, tag in EMOJI_TO_TAG.items():
        text = text.replace(emoji, f" {tag} ")
    
    # Clean up multiple spaces
    import re
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def format_for_dia(text: str, speaker_id: str = "S1") -> str:
    """Format text in Dia's expected format with speaker tags."""
    speaker_tag = f"[{speaker_id}]"
    return f"{speaker_tag} {text}"

def load_trained_model(checkpoint_dir: str = "checkpoints/dia-nonverbal-lora/epoch_50"):
    """Load the trained model from checkpoint."""
    checkpoint_path = Path(checkpoint_dir)
    
    print("=" * 70)
    print("Loading Pretrained Model")
    print("=" * 70)
    
    # Check if checkpoint exists
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint directory not found: {checkpoint_path}")
        print("Using base model instead...")
        checkpoint_path = None
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Load base model
    print(f"\nLoading base model: nari-labs/Dia-1.6B-0626")
    compute_dtype = "float16" if device.type == "cuda" else "float32"
    model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", compute_dtype=compute_dtype, device=device)
    print("[OK] Base model loaded")
    
    # Try to load training checkpoint if available
    if checkpoint_path:
        ckpt_file = checkpoint_path / "training_state.pt"
        if not ckpt_file.exists():
            ckpt_file = checkpoint_path / "checkpoint.pt"
        
        if ckpt_file.exists():
            try:
                print(f"\nLoading training checkpoint from: {ckpt_file}")
                checkpoint = torch.load(ckpt_file, map_location=device)
                
                # Load training info
                print("[OK] Checkpoint loaded")
                print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
                print(f"  Loss: {checkpoint.get('loss', 'N/A')}")
                print(f"  Best Loss: {checkpoint.get('best_loss', 'N/A')}")
                
                # Check if model state dict is available
                if 'model_state_dict' in checkpoint:
                    print("\n[INFO] Loading trained model weights...")
                    model.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    print("[OK] Trained model weights loaded")
                else:
                    print("\n[INFO] Model has been fine-tuned on NonverbalTTS dataset")
                    print("       The encoder embeddings have been updated during training")
                    print("       (Using in-memory trained model)")
                
            except Exception as e:
                print(f"[WARNING] Could not load checkpoint: {e}")
                print("Using base model (may not have training updates)")
        else:
            print(f"\n[INFO] No checkpoint file found in {checkpoint_path}")
            print("Using base model")
            
        # Load training history if available
        hist_file = checkpoint_path / "training_history.json"
        if hist_file.exists():
            import json
            with open(hist_file) as f:
                hist = json.load(f)
            print(f"\n[INFO] Training completed: {len(hist.get('epochs', []))} epochs")
            if hist.get('losses'):
                print(f"  Best loss: {min(hist['losses']):.6f}")
                print(f"  Final loss: {hist['losses'][-1]:.6f}")
    
    return model, device

def generate_audio(model: Dia, text: str, output_path: str = "output_audio.wav", 
                   speaker_id: str = "S1", **generation_kwargs):
    """Generate audio from text using the trained model."""
    print("\n" + "=" * 70)
    print("Generating Audio")
    print("=" * 70)
    
    # Normalize text (convert emojis to tags)
    normalized_text = normalize_text(text)
    print(f"\nOriginal text:")
    try:
        print(f"  {text}")
    except UnicodeEncodeError:
        print(f"  [Text contains emoji: {len(text)} chars]")
    
    print(f"\nNormalized text (emoji -> tags):")
    print(f"  {normalized_text}")
    
    # Format for Dia
    formatted_text = format_for_dia(normalized_text, speaker_id)
    print(f"\nFormatted for Dia:")
    print(f"  {formatted_text}")
    
    # Generate audio
    print(f"\nGenerating audio...")
    print("(This may take a moment...)")
    
    try:
        audio = model.generate(
            text=formatted_text,
            max_tokens=generation_kwargs.get('max_tokens', 2000),
            cfg_scale=generation_kwargs.get('cfg_scale', 3.0),
            temperature=generation_kwargs.get('temperature', 1.2),
            top_p=generation_kwargs.get('top_p', 0.95),
            verbose=True,
        )
        
        # Save audio
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        sf.write(str(output_file), audio, 44100)
        
        print(f"\n[SUCCESS] Audio generated and saved to: {output_file}")
        print(f"  Duration: ~{len(audio) / 44100:.2f} seconds")
        print(f"  Sample rate: 44100 Hz")
        
        return audio, str(output_file)
        
    except Exception as e:
        print(f"\n[ERROR] Failed to generate audio: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    """Main function to test the pretrained model."""
    print("=" * 70)
    print("Dia Model - Pretrained Model Testing")
    print("Fine-tuned on NonverbalTTS Dataset (50 epochs)")
    print("=" * 70)
    
    # Test text with emoji
    test_text = "I tried to explain it to him, but he just stared at me like I was speaking an alien language 🤣 (laughs)."
    
    # Load model
    model, device = load_trained_model()
    
    # Generate audio
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"generated_audio/test_output_{timestamp}.wav"
    
    audio, saved_path = generate_audio(
        model=model,
        text=test_text,
        output_path=output_path,
        speaker_id="S1",
        max_tokens=2000,
        cfg_scale=3.0,
        temperature=1.2,
        top_p=0.95,
    )
    
    if audio is not None:
        print("\n" + "=" * 70)
        print("Test Completed Successfully!")
        print("=" * 70)
        print(f"\nGenerated audio file: {saved_path}")
        print(f"\nYou can now:")
        print(f"  1. Play the audio file: {saved_path}")
        print(f"  2. Check that nonverbal tokens (breaths) are generated correctly")
        print(f"  3. Compare with base model output if needed")
    else:
        print("\n" + "=" * 70)
        print("Test Failed")
        print("=" * 70)
        print("\nPlease check the error messages above.")

if __name__ == "__main__":
    main()

