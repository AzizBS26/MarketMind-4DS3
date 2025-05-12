import os
import argparse
import torch
import soundfile as sf
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

from src.model import TextToSpeechModel


def generate_speech(args):
    """
    Generate speech from text using the trained model
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = TextToSpeechModel(
        vocab_size=args.vocab_size,
        embedding_dim=args.embedding_dim,
        encoder_layers=args.encoder_layers,
        encoder_heads=args.encoder_heads,
        encoder_dim=args.encoder_dim,
        duration_hidden=args.duration_hidden,
        decoder_dim=args.decoder_dim,
        decoder_layers=args.decoder_layers,
        decoder_heads=args.decoder_heads,
        decoder_ffn_dim=args.decoder_ffn_dim,
        postnet_dim=args.postnet_dim,
        n_mels=args.n_mels,
        sample_rate=args.sample_rate,
        dropout=0.0  # No dropout for inference
    )
    
    # Load model weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Generate speech
    print(f"Generating speech for text: {args.text}")
    
    with torch.no_grad():
        # Generate audio
        waveform, mel_spectrogram = model.generate(args.text, alpha=args.speed)
        
        # Convert to numpy
        waveform = waveform[0].cpu().numpy()
        mel_spectrogram = mel_spectrogram[0].cpu().numpy()
        
        # Save audio
        sf.write(args.output, waveform, args.sample_rate)
        print(f"Audio saved to {args.output}")
        
        # Plot and save mel spectrogram if requested
        if args.plot:
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(
                mel_spectrogram,
                sr=args.sample_rate,
                hop_length=256,
                x_axis='time',
                y_axis='mel',
                fmin=0,
                fmax=8000
            )
            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel spectrogram')
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.splitext(args.output)[0] + '_mel.png'
            plt.savefig(plot_path)
            print(f"Mel spectrogram saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate speech from text")
    
    # Input/output parameters
    parser.add_argument("--text", type=str, required=True,
                        help="Text to synthesize")
    parser.add_argument("--output", type=str, default="output.wav",
                        help="Path to save output audio")
    parser.add_argument("--model_path", type=str, default="output/model.pt",
                        help="Path to model weights")
    parser.add_argument("--plot", action="store_true",
                        help="Plot and save mel spectrogram")
    
    # Generation parameters
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speech speed factor (lower is faster)")
    
    # Model parameters (should match training parameters)
    parser.add_argument("--vocab_size", type=int, default=100, help="Vocabulary size")
    parser.add_argument("--embedding_dim", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--encoder_layers", type=int, default=4, help="Number of encoder layers")
    parser.add_argument("--encoder_heads", type=int, default=8, help="Number of encoder attention heads")
    parser.add_argument("--encoder_dim", type=int, default=512, help="Encoder hidden dimension")
    parser.add_argument("--duration_hidden", type=int, default=256, help="Duration predictor hidden dimension")
    parser.add_argument("--decoder_dim", type=int, default=512, help="Decoder hidden dimension")
    parser.add_argument("--decoder_layers", type=int, default=6, help="Number of decoder layers")
    parser.add_argument("--decoder_heads", type=int, default=8, help="Number of decoder attention heads")
    parser.add_argument("--decoder_ffn_dim", type=int, default=1024, help="Decoder FFN dimension")
    parser.add_argument("--postnet_dim", type=int, default=512, help="PostNet dimension")
    parser.add_argument("--n_mels", type=int, default=80, help="Number of mel spectrogram bins")
    parser.add_argument("--sample_rate", type=int, default=22050, help="Audio sample rate")
    
    args = parser.parse_args()
    
    generate_speech(args) 