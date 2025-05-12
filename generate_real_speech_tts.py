import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import soundfile as sf
import librosa
from TTS.api import TTS

def main():
    parser = argparse.ArgumentParser(description="Generate real speech using TTS API")
    parser.add_argument("--text", type=str, default="Hello, this is a test of the text to speech system.",
                      help="Text to synthesize")
    parser.add_argument("--output_dir", type=str, default="output",
                      help="Directory to save output files")
    parser.add_argument("--model", type=str, default="tts_models/en/ljspeech/tacotron2-DDC",
                      help="Model to use for TTS")
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize TTS
    print(f"Initializing TTS with model {args.model}...")
    tts = TTS(args.model)
    
    # Generate speech
    print(f"Generating speech for text: '{args.text}'")
    output_path = os.path.join(args.output_dir, "real_speech_tts.wav")
    tts.tts_to_file(text=args.text, file_path=output_path)
    print(f"Audio saved to {output_path}")
    
    # Load audio for visualization
    waveform, sample_rate = librosa.load(output_path, sr=None)
    
    # Create and save spectrogram
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(waveform)), ref=np.max)
    plt.imshow(D, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Generated Speech Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "real_speech_tts_spectrogram.png"))
    print(f"Spectrogram saved to {os.path.join(args.output_dir, 'real_speech_tts_spectrogram.png')}")
    
    # List available models
    print("\nAvailable TTS models:")
    print("---------------------")
    for model in tts.list_models():
        if "en" in model and "tts" in model:
            print(f"- {model}")

if __name__ == "__main__":
    main() 