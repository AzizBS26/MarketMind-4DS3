import torch
import torch.nn as nn
import torchaudio
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
import argparse
from tqdm import tqdm

class SimpleEncoder(nn.Module):
    """
    Simple character-level encoder
    """
    def __init__(self, input_dim=128, hidden_dim=512, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True
        )
        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        return self.linear(x)

class Decoder(nn.Module):
    """
    Simple mel spectrogram decoder
    """
    def __init__(self, input_dim=512, hidden_dim=512, n_mels=80):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_dim, n_mels)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        return self.linear(x)

class Vocoder(nn.Module):
    """
    Simple vocoder to convert mel spectrograms to waveforms
    """
    def __init__(self, n_mels=80, channels=512):
        super().__init__()
        
        self.conv1 = nn.ConvTranspose1d(n_mels, channels, kernel_size=4, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.ConvTranspose1d(channels, channels // 2, kernel_size=4, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.ConvTranspose1d(channels // 2, channels // 4, kernel_size=4, stride=2, padding=1)
        self.relu3 = nn.ReLU()
        
        self.conv4 = nn.ConvTranspose1d(channels // 4, channels // 8, kernel_size=4, stride=2, padding=1)
        self.relu4 = nn.ReLU()
        
        self.conv5 = nn.ConvTranspose1d(channels // 8, 1, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        # Input: [batch_size, n_mels, seq_len]
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.tanh(self.conv5(x))
        return x

class SimpleTTS(nn.Module):
    """
    Simple end-to-end TTS model
    """
    def __init__(self, vocab_size=128, encoder_dim=512, decoder_dim=512, n_mels=80):
        super().__init__()
        self.encoder = SimpleEncoder(vocab_size, encoder_dim)
        self.decoder = Decoder(encoder_dim, decoder_dim, n_mels)
        self.vocoder = Vocoder(n_mels)
        
    def forward(self, text_indices):
        # Encode text
        encoded = self.encoder(text_indices)
        
        # Decode to mel spectrogram
        mel_output = self.decoder(encoded)
        
        # Convert to waveform
        waveform = self.vocoder(mel_output.transpose(1, 2))
        
        return {
            'mel_output': mel_output,
            'waveform': waveform
        }
    
    def generate(self, text):
        """
        Generate speech from text
        """
        # Convert text to indices
        chars = [ord(c) for c in text.lower() if ord(c) < 128]
        text_indices = torch.tensor(chars).unsqueeze(0)  # [1, seq_len]
        
        # Generate
        with torch.no_grad():
            outputs = self(text_indices)
            
        # Get output
        mel = outputs['mel_output'][0]  # [seq_len, n_mels]
        waveform = outputs['waveform'][0, 0]  # [seq_len]
        
        return waveform, mel


def test_generation():
    """
    Test speech generation
    """
    # Create model
    model = SimpleTTS()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Generate speech
    text = "Hello, this is a test of the text to speech model."
    waveform, mel = model.generate(text)
    
    # Convert to numpy
    waveform = waveform.cpu().numpy()
    mel = mel.cpu().numpy()
    
    # Save audio
    os.makedirs('output', exist_ok=True)
    sf.write('output/test.wav', waveform, 22050)
    
    # Plot mel spectrogram
    plt.figure(figsize=(10, 4))
    plt.imshow(mel.T, aspect='auto', origin='lower')
    plt.tight_layout()
    plt.savefig('output/mel_spectrogram.png')
    plt.close()
    
    print(f"Generated audio saved to output/test.wav")
    print(f"Mel spectrogram saved to output/mel_spectrogram.png")


if __name__ == "__main__":
    test_generation() 