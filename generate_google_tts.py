import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import librosa
from gtts import gTTS

def main():
    parser = argparse.ArgumentParser(description="Generate speech using Google Text-to-Speech")
    parser.add_argument("--text", type=str, default="Hello, this is a test of the text to speech system.",
                      help="Text to synthesize")
    parser.add_argument("--output_dir", type=str, default="output",
                      help="Directory to save output files")
    parser.add_argument("--lang", type=str, default="en",
                      help="Language code (e.g., 'en', 'fr', 'es')")
    parser.add_argument("--slow", action="store_true",
                      help="Generate slower speech")
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate speech
    print(f"Generating speech for text: '{args.text}'")
    output_path = os.path.join(args.output_dir, "google_tts.mp3")
    
    # Create gTTS object
    tts = gTTS(text=args.text, lang=args.lang, slow=args.slow)
    
    # Save to file
    tts.save(output_path)
    print(f"Audio saved to {output_path}")
    
    try:
        # Convert to WAV for better compatibility
        wav_path = os.path.join(args.output_dir, "google_tts.wav")
        import librosa
        y, sr = librosa.load(output_path)
        sf.write(wav_path, y, sr)
        print(f"WAV audio saved to {wav_path}")
        
        # Create and save spectrogram
        plt.figure(figsize=(10, 4))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        plt.imshow(D, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Generated Speech Spectrogram')
        plt.xlabel('Time')
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()
        spec_path = os.path.join(args.output_dir, "google_tts_spectrogram.png")
        plt.savefig(spec_path)
        print(f"Spectrogram saved to {spec_path}")
    except Exception as e:
        print(f"Warning: Could not create spectrogram or WAV file: {e}")
        print("MP3 file was still generated successfully.")

if __name__ == "__main__":
    main() 