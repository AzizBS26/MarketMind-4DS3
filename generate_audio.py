import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchaudio
import os
from src.model import TextToSpeechModel
from src.text_encoder import PhonemeEmbedding
from src.utils import text_to_phonemes

def main():
    parser = argparse.ArgumentParser(description='Generate speech from text using trained TTS model')
    parser.add_argument('--text', type=str, default="Hello, this is a test.", help='Text to synthesize')
    parser.add_argument('--model_path', type=str, default="output/model.pt", help='Path to the trained model')
    parser.add_argument('--output_dir', type=str, default="output", help='Directory to save output files')
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parameters should match those used in training
    vocab_size = 100  # Adjust if needed
    embedding_dim = 256
    encoder_layers = 3
    encoder_heads = 2
    encoder_dim = 256
    encoder_ff_dim = 1024
    duration_predictor_filters = 256
    duration_predictor_kernel_size = 3
    decoder_layers = 4
    decoder_dim = 256
    
    # Create model
    model = TextToSpeechModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        encoder_layers=encoder_layers,
        encoder_heads=encoder_heads,
        encoder_dim=encoder_dim,
        encoder_ff_dim=encoder_ff_dim,
        duration_predictor_filters=duration_predictor_filters,
        duration_predictor_kernel_size=duration_predictor_kernel_size,
        decoder_layers=decoder_layers,
        decoder_dim=decoder_dim
    )
    
    # Load model weights
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # Convert text to phonemes
    phoneme_seq = text_to_phonemes(args.text)
    print(f"Phonemes: {phoneme_seq}")
    
    # Create phoneme embedding
    phoneme_embedding = PhonemeEmbedding(vocab_size, embedding_dim)
    phoneme_indices = phoneme_embedding.convert_phonemes_to_indices(phoneme_seq)
    
    # Convert to tensor
    phoneme_tensor = torch.tensor(phoneme_indices).unsqueeze(0)  # Add batch dimension
    
    # Generate speech
    with torch.no_grad():
        outputs = model(phoneme_tensor)
        mel_output = outputs["mel_output"]
        mel_postnet = outputs["mel_postnet_output"]
        
        # Generate audio using vocoder
        waveform = model.vocoder_inference(mel_postnet)
    
    # Save the audio file
    output_path = os.path.join(args.output_dir, "generated.wav")
    torchaudio.save(output_path, waveform, 22050)
    print(f"Audio saved to {output_path}")
    
    # Save mel spectrogram for visualization
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_postnet[0].detach().cpu().numpy(), aspect='auto', origin='lower')
    plt.colorbar()
    plt.title('Generated Mel Spectrogram')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "generated_mel.png"))
    print(f"Mel spectrogram saved to {os.path.join(args.output_dir, 'generated_mel.png')}")

if __name__ == "__main__":
    main() 