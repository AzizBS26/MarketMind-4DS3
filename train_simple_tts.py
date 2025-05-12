import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
import argparse
from tqdm import tqdm
from test_tts import SimpleTTS

class SimpleAudioDataset(Dataset):
    """
    Simple audio dataset for TTS training
    """
    def __init__(self, audio_dir, max_samples=10):
        super().__init__()
        self.samples = []
        
        # Simple data - just create some text-audio pairs
        for i in range(max_samples):
            self.samples.append({
                'text': f"This is sample number {i}",
                'audio_length': 3000 + i * 1000  # Dummy length
            })
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample['text']
        
        # Convert text to indices
        chars = [ord(c) for c in text.lower() if ord(c) < 128]
        text_indices = torch.tensor(chars)
        
        # Create dummy audio tensor
        audio_length = sample['audio_length']
        audio = torch.randn(audio_length)
        
        # Create dummy mel spectrogram
        n_mels = 80
        mel_length = audio_length // 256  # Approximate reduction factor
        mel = torch.randn(n_mels, mel_length)
        
        return {
            'text': text,
            'text_indices': text_indices,
            'text_length': len(text_indices),
            'audio': audio,
            'audio_length': audio_length,
            'mel': mel,
            'mel_length': mel_length
        }

def collate_fn(batch):
    """
    Collate function for DataLoader
    """
    # Get max lengths
    max_text_len = max(item['text_length'] for item in batch)
    max_audio_len = max(item['audio_length'] for item in batch)
    max_mel_len = max(item['mel_length'] for item in batch)
    
    # Initialize tensors
    text_indices = torch.zeros(len(batch), max_text_len, dtype=torch.long)
    audio = torch.zeros(len(batch), max_audio_len)
    mel = torch.zeros(len(batch), 80, max_mel_len)
    
    # Fill tensors
    texts = []
    text_lengths = []
    audio_lengths = []
    mel_lengths = []
    
    for i, item in enumerate(batch):
        texts.append(item['text'])
        
        # Text
        text_len = item['text_length']
        text_indices[i, :text_len] = item['text_indices']
        text_lengths.append(text_len)
        
        # Audio
        audio_len = item['audio_length']
        audio[i, :audio_len] = item['audio']
        audio_lengths.append(audio_len)
        
        # Mel
        mel_len = item['mel_length']
        mel[i, :, :mel_len] = item['mel']
        mel_lengths.append(mel_len)
    
    return {
        'texts': texts,
        'text_indices': text_indices,
        'text_lengths': torch.tensor(text_lengths),
        'audio': audio,
        'audio_lengths': torch.tensor(audio_lengths),
        'mel': mel,
        'mel_lengths': torch.tensor(mel_lengths)
    }

class TTSLoss(nn.Module):
    """
    Loss function for TTS training
    """
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, outputs, targets):
        mel_output = outputs['mel_output']  # [batch_size, seq_len, n_mels]
        waveform = outputs['waveform'].squeeze(1)  # [batch_size, time]
        
        mel_target = targets['mel']  # [batch_size, n_mels, seq_len]
        audio_target = targets['audio']  # [batch_size, time]
        
        # Get the dimensions
        batch_size, out_seq_len, n_mels = mel_output.shape
        _, target_n_mels, target_seq_len = mel_target.shape
        
        # We need to make the sizes match for loss calculation
        # First transpose the output to match target format [batch_size, n_mels, seq_len]
        mel_output = mel_output.transpose(1, 2)
        
        # Then use a mask to calculate loss only on overlapping parts
        min_seq_len = min(out_seq_len, target_seq_len)
        
        # Calculate masked loss
        mel_loss = self.l1_loss(
            mel_output[:, :, :min_seq_len], 
            mel_target[:, :, :min_seq_len]
        )
        
        # For waveform loss, also use the minimum length
        min_audio_len = min(waveform.size(-1), audio_target.size(-1))
        wave_loss = self.mse_loss(
            waveform[:, :min_audio_len], 
            audio_target[:, :min_audio_len]
        )
        
        # Total loss
        total_loss = mel_loss + wave_loss
        
        return total_loss, {
            'mel_loss': mel_loss.item(),
            'wave_loss': wave_loss.item(),
            'total_loss': total_loss.item()
        }

def train(args):
    """
    Train the TTS model
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dataset and dataloader
    dataset = SimpleAudioDataset(args.data_dir, max_samples=args.max_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Create model
    model = SimpleTTS(
        vocab_size=128,
        encoder_dim=args.encoder_dim,
        decoder_dim=args.decoder_dim,
        n_mels=80
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create loss function
    criterion = TTSLoss()
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        # Progress bar
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in progress_bar:
            # Move data to device
            text_indices = batch['text_indices'].to(device)
            mel_target = batch['mel'].to(device)
            audio_target = batch['audio'].to(device)
            
            # Forward pass
            outputs = model(text_indices)
            
            # Calculate loss
            loss, losses_dict = criterion(outputs, {'mel': mel_target, 'audio': audio_target})
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            progress_bar.set_postfix(loss=loss.item())
            epoch_loss += loss.item()
        
        # Calculate average epoch loss
        epoch_loss /= len(dataloader)
        print(f"Epoch {epoch+1} loss: {epoch_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss
        }, checkpoint_path)
    
    # Save final model
    model_path = os.path.join(args.output_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Generate sample
    model.eval()
    with torch.no_grad():
        text = "This is a sample generated after training."
        waveform, mel = model.generate(text)
        
        # Convert to numpy
        waveform = waveform.cpu().numpy()
        mel = mel.cpu().numpy()
        
        # Save audio
        sample_path = os.path.join(args.output_dir, "sample.wav")
        import soundfile as sf
        sf.write(sample_path, waveform, 22050)
        
        # Plot mel spectrogram
        plt.figure(figsize=(10, 4))
        plt.imshow(mel.T, aspect='auto', origin='lower')
        plt.title("Generated Mel Spectrogram")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "sample_mel.png"))
        plt.close()
    
    print(f"Sample generated and saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a simple TTS model")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing audio data")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save outputs")
    parser.add_argument("--max_samples", type=int, default=10, help="Maximum number of samples to use")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    
    # Model parameters
    parser.add_argument("--encoder_dim", type=int, default=256, help="Encoder dimension")
    parser.add_argument("--decoder_dim", type=int, default=256, help="Decoder dimension")
    
    args = parser.parse_args()
    
    train(args) 