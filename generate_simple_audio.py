import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os
from test_tts import SimpleTTS

def main():
    parser = argparse.ArgumentParser(description='Generate speech from text using SimpleTTS model')
    parser.add_argument('--text', type=str, default="Hello, this is a test of the text to speech system.", help='Text to synthesize')
    parser.add_argument('--model_path', type=str, default="output/model.pt", help='Path to the trained model')
    parser.add_argument('--output_dir', type=str, default="output", help='Directory to save output files')
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create model
    model = SimpleTTS(
        vocab_size=128,
        encoder_dim=256,
        decoder_dim=256,
        n_mels=80
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model weights if the file exists
    if os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        print(f"Model path {args.model_path} not found. Using untrained model.")
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Generate speech
    print(f"Generating speech for text: '{args.text}'")
    waveform, mel = model.generate(args.text)
    
    # Convert to numpy
    waveform = waveform.cpu().numpy()
    mel = mel.cpu().numpy()
    
    # Save audio
    output_wav = os.path.join(args.output_dir, "generated_simple.wav")
    sf.write(output_wav, waveform, 22050)
    print(f"Audio saved to {output_wav}")
    
    # Plot mel spectrogram
    plt.figure(figsize=(10, 4))
    plt.imshow(mel.T, aspect='auto', origin='lower')
    plt.title("Generated Mel Spectrogram")
    plt.tight_layout()
    output_mel = os.path.join(args.output_dir, "generated_simple_mel.png")
    plt.savefig(output_mel)
    plt.close()
    print(f"Mel spectrogram saved to {output_mel}")

if __name__ == "__main__":
    main() 