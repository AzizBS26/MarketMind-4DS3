import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pyttsx3
import wave
import soundfile as sf
import tempfile
import array
import struct

def main():
    parser = argparse.ArgumentParser(description="Generate speech using pyttsx3 (offline TTS)")
    parser.add_argument("--text", type=str, default="Hello, this is a test of the offline text to speech system.",
                      help="Text to synthesize")
    parser.add_argument("--output_dir", type=str, default="output",
                      help="Directory to save output files")
    parser.add_argument("--rate", type=int, default=150,
                      help="Speech rate (words per minute)")
    parser.add_argument("--volume", type=float, default=1.0,
                      help="Volume (0.0 to 1.0)")
    parser.add_argument("--voice_id", type=int, default=None,
                      help="Voice ID to use (show available voices)")
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the TTS engine
    engine = pyttsx3.init()
    
    # Set properties
    engine.setProperty('rate', args.rate)    # Speed of speech
    engine.setProperty('volume', args.volume)  # Volume (0.0 to 1.0)
    
    # Get available voices
    voices = engine.getProperty('voices')
    print("Available voices:")
    for i, voice in enumerate(voices):
        print(f"{i}: {voice.name} ({voice.id})")
    
    # Set voice
    if args.voice_id is not None and 0 <= args.voice_id < len(voices):
        voice_idx = args.voice_id
    else:
        # Try to find an English voice, default to first voice
        voice_idx = 0
        for i, voice in enumerate(voices):
            if "en" in voice.id.lower():
                voice_idx = i
                break
    
    # Use selected voice
    engine.setProperty('voice', voices[voice_idx].id)
    print(f"Using voice: {voices[voice_idx].name}")
    
    print(f"Generating speech for text: '{args.text}'")
    
    # Save to a temporary WAV file
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_file.close()
    
    # Use pyttsx3 to generate speech and save to file
    def on_complete(name):
        print(f"Speech generation complete: {name}")
    
    engine.connect('finished-utterance', on_complete)
    engine.save_to_file(args.text, temp_file.name)
    engine.runAndWait()
    
    # Copy to final output location
    output_path = os.path.join(args.output_dir, "pyttsx3_speech.wav")
    with wave.open(temp_file.name, 'rb') as wf:
        # Get parameters
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        
        # Convert frames to numpy array for spectrogram
        if width == 2:  # 16-bit audio
            data = np.frombuffer(frames, dtype=np.int16)
        else:  # 8-bit audio
            data = np.frombuffer(frames, dtype=np.uint8) - 128
        
        # Save to output file
        sf.write(output_path, data, rate)
    
    # Clean up temporary file
    os.unlink(temp_file.name)
    
    print(f"Audio saved to {output_path}")
    
    # Create spectrogram
    try:
        # Convert to mono if stereo
        if channels == 2:
            data = data.reshape(-1, 2).mean(axis=1)
        
        # Ensure data is not empty
        if len(data) == 0:
            raise ValueError("Audio data is empty")
        
        # Add small offset to avoid log(0)
        NFFT = 512
        noverlap = NFFT // 2
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 4))
        spec = ax.specgram(data, NFFT=NFFT, Fs=rate, noverlap=noverlap, cmap='viridis', 
                          scale='dB', vmin=-100, vmax=20)[0]
        
        ax.set_title('Generated Speech Spectrogram')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        
        # Add colorbar
        cbar = fig.colorbar(ax.collections[0] if ax.collections else plt.cm.ScalarMappable(cmap='viridis'), 
                           ax=ax, format='%+2.0f dB')
        cbar.set_label('Intensity (dB)')
        
        fig.tight_layout()
        
        spec_path = os.path.join(args.output_dir, "pyttsx3_spectrogram.png")
        plt.savefig(spec_path)
        plt.close()
        print(f"Spectrogram saved to {spec_path}")
    except Exception as e:
        print(f"Warning: Could not create spectrogram: {e}")
        print("Audio file was still generated successfully.")

if __name__ == "__main__":
    main() 