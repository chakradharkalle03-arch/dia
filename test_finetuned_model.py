"""
Test the fine-tuned SpeechT5 or XTTS model.
"""
import torch
from pathlib import Path
import argparse
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

def test_speecht5(checkpoint_path: str, text: str, output_file: str = "test_output.wav"):
    """Test fine-tuned SpeechT5 model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 70)
    print("Testing Fine-tuned SpeechT5 Model")
    print("=" * 70)
    print(f"\nDevice: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Text: {text}")
    
    # Load model
    print("\nLoading model...")
    processor = SpeechT5Processor.from_pretrained(checkpoint_path)
    model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint_path)
    model.to(device)
    model.eval()
    
    # Load vocoder
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    vocoder.to(device)
    
    print("[OK] Model loaded")
    
    # Process text
    print("\nGenerating speech...")
    inputs = processor(text=text, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        speech = model.generate_speech(
            inputs["input_ids"],
            vocoder,
            speaker_embeddings=None
        )
    
    # Save audio
    import soundfile as sf
    sf.write(output_file, speech.cpu().numpy(), samplerate=16000)
    
    print(f"[OK] Audio saved to: {output_file}")
    print(f"     Duration: {len(speech) / 16000:.2f} seconds")


def test_xtts(checkpoint_path: str, text: str, output_file: str = "test_output.wav"):
    """Test fine-tuned XTTS model."""
    try:
        from TTS.api import TTS
    except ImportError:
        print("Error: TTS library not installed")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 70)
    print("Testing Fine-tuned XTTS Model")
    print("=" * 70)
    print(f"\nDevice: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Text: {text}")
    
    # Load model
    print("\nLoading model...")
    tts = TTS(model_path=checkpoint_path)
    tts.to(device)
    print("[OK] Model loaded")
    
    # Generate
    print("\nGenerating speech...")
    tts.tts_to_file(
        text=text,
        file_path=output_file
    )
    
    print(f"[OK] Audio saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Test fine-tuned TTS model")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--model_type", type=str, choices=["speecht5", "xtts"], default="speecht5",
                       help="Type of model (speecht5 or xtts)")
    parser.add_argument("--text", type=str,
                       default="[S1] Hello, this is a test of the fine-tuned model. (laughs) It can generate nonverbal sounds!",
                       help="Text to synthesize")
    parser.add_argument("--output", type=str, default="test_output.wav",
                       help="Output audio file")
    
    args = parser.parse_args()
    
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return
    
    if args.model_type == "speecht5":
        test_speecht5(args.checkpoint, args.text, args.output)
    elif args.model_type == "xtts":
        test_xtts(args.checkpoint, args.text, args.output)


if __name__ == "__main__":
    main()

