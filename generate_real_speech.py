import torch
import soundfile as sf
import matplotlib.pyplot as plt
import os
import argparse
from transformers import AutoProcessor, AutoModel

def main():
    parser = argparse.ArgumentParser(description="Generate real speech with pre-trained TTS model")
    parser.add_argument("--text", type=str, default="Hello, this is a test of the text to speech system.",
                      help="Text to synthesize")
    parser.add_argument("--output_dir", type=str, default="output",
                      help="Directory to save output files")
    parser.add_argument("--speaker_id", type=int, default=0,
                      help="Speaker ID for multi-speaker models")
    parser.add_argument("--model", type=str, default="suno/bark-small",
                      help="Model name or path from HuggingFace")
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print(f"Loading model {args.model}...")
    
    # Load TTS model and processor
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)
    
    print(f"Generating speech for text: '{args.text}'")
    
    # Generate speech
    inputs = processor(
        text=[args.text],
        return_tensors="pt",
    ).to(device)
    
    # Generate audio
    with torch.no_grad():
        output = model.generate(**inputs, do_sample=True)
    
    # Convert to waveform
    speech = output[0].cpu().numpy()
    
    # Save audio output
    output_path = os.path.join(args.output_dir, "real_speech.wav")
    sample_rate = model.generation_config.sample_rate
    sf.write(output_path, speech, sample_rate)
    print(f"Audio saved to {output_path}")
    
    # Create spectrogram for visualization
    plt.figure(figsize=(10, 4))
    plt.specgram(speech, Fs=sample_rate, cmap="viridis")
    plt.title("Generated Speech Spectrogram")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "real_speech_spectrogram.png"))
    print(f"Spectrogram saved to {os.path.join(args.output_dir, 'real_speech_spectrogram.png')}")

if __name__ == "__main__":
    main() 