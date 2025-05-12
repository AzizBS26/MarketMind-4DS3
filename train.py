import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import time
import random
import json

from src.model import TextToSpeechModel

# Add a model adapter to handle the dataset format
class ModelAdapter(nn.Module):
    """
    Adapter for the TextToSpeechModel to handle dataset format
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, text_indices, text_lengths=None):
        """
        Adapt the input format to the model's expected format
        """
        # Convert tensor indices to strings of characters
        batch_size = text_indices.size(0)
        phoneme_sequences = []
        
        for i in range(batch_size):
            # Get valid indices (non-padding)
            if text_lengths is not None:
                length = text_lengths[i]
                indices = text_indices[i, :length]
            else:
                # Use all non-zero indices
                indices = text_indices[i, text_indices[i] != 0]
            
            # Convert to string of characters (our simple encoding)
            chars = "abcdefghijklmnopqrstuvwxyz0123456789 ,.!?-'"
            phoneme = ''.join([chars[idx-1] if 0 < idx <= len(chars) else '' for idx in indices])
            phoneme_sequences.append(phoneme)
        
        # Forward pass through the model
        return self.model(phoneme_sequences)

class LibriSpeechDataset(Dataset):
    """
    Dataset for preprocessed LibriSpeech data
    """
    def __init__(self, metadata_path, max_samples=None):
        super().__init__()
        
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # Limit samples if needed
        if max_samples is not None and max_samples > 0:
            self.metadata = self.metadata[:max_samples]
        
        print(f"Loaded dataset with {len(self.metadata)} samples")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        # Load waveform and mel spectrogram
        waveform = torch.load(item['waveform_path'])
        mel = torch.load(item['mel_path'])
        
        # Get text
        text = item['text']
        
        # Simple character-based encoding
        # Create a mapping from characters to indices
        chars = set("abcdefghijklmnopqrstuvwxyz0123456789 ,.!?-'")
        char_to_idx = {c: i+1 for i, c in enumerate(sorted(chars))}  # Start from 1, 0 is for padding
        
        # Convert text to indices
        text_indices = [char_to_idx.get(c.lower(), 0) for c in item['text'] if c.lower() in chars]
        text_indices = torch.tensor(text_indices, dtype=torch.long)
        
        return {
            'text': text,
            'text_indices': text_indices,
            'text_length': len(text_indices),
            'waveform': waveform,
            'waveform_length': item['waveform_length'],
            'mel': mel,
            'mel_length': item['spec_length']
        }


def collate_fn(batch):
    """
    Collate function for the dataloader
    """
    # Get max lengths
    max_text_len = max(item['text_length'] for item in batch)
    max_waveform_len = max(item['waveform_length'] for item in batch)
    max_mel_len = max(item['mel_length'] for item in batch)
    
    # Initialize tensors
    batch_size = len(batch)
    text_indices = torch.zeros(batch_size, max_text_len, dtype=torch.long)
    waveforms = torch.zeros(batch_size, max_waveform_len)
    mels = torch.zeros(batch_size, 80, max_mel_len)
    
    # Fill tensors
    texts = []
    text_lengths = []
    waveform_lengths = []
    mel_lengths = []
    
    for i, item in enumerate(batch):
        texts.append(item['text'])
        
        # Text
        text_len = item['text_length']
        text_indices[i, :text_len] = item['text_indices']
        text_lengths.append(text_len)
        
        # Waveform
        waveform_len = item['waveform_length']
        waveforms[i, :waveform_len] = item['waveform']
        waveform_lengths.append(waveform_len)
        
        # Mel
        mel_len = item['mel_length']
        mels[i, :, :mel_len] = item['mel']
        mel_lengths.append(mel_len)
    
    return {
        'texts': texts,
        'text_indices': text_indices,
        'text_lengths': torch.tensor(text_lengths),
        'waveforms': waveforms,
        'waveform_lengths': torch.tensor(waveform_lengths),
        'mels': mels,
        'mel_lengths': torch.tensor(mel_lengths)
    }


class TTSLoss(nn.Module):
    """
    Loss function for TTS training
    """
    def __init__(self, mel_weight=1.0, duration_weight=0.1, postnet_weight=1.0, waveform_weight=0.5):
        super().__init__()
        self.mel_weight = mel_weight
        self.duration_weight = duration_weight
        self.postnet_weight = postnet_weight
        self.waveform_weight = waveform_weight
        
        self.mse_loss = nn.MSELoss(reduction='none')
        self.l1_loss = nn.L1Loss(reduction='none')
        self.mae_loss = nn.L1Loss()
    
    def forward(self, outputs, targets):
        # Get outputs
        mel_output = outputs['mel_output']  # [B, T1, D]
        postnet_output = outputs['postnet_output']  # [B, T1, D]
        duration_preds = outputs['duration_preds']
        waveform = outputs['waveform']  # [B, 1, T3]
        
        # Get targets
        mel_target = targets['mel_spectrograms']  # [B, D, T2]
        waveform_target = targets['waveforms']  # [B, T4]
        spec_lengths = targets['spectrogram_lengths']
        
        # Handle dimension discrepancy - transpose mel_target to match output shape
        mel_target = mel_target.transpose(1, 2)  # Now [B, T2, D]
        
        # We don't have ground truth durations, so we'll skip that loss component
        
        # Get the minimum length for both mel spectrograms
        B = mel_output.size(0)
        D = mel_output.size(2)
        T_out = mel_output.size(1)
        T_target = mel_target.size(1)
        min_t = min(T_out, T_target)
        
        # Truncate to the minimum length
        mel_output = mel_output[:, :min_t, :]
        mel_target = mel_target[:, :min_t, :]
        
        # Create mel mask
        mel_mask = torch.ones(B, min_t, 1).to(mel_output.device)
        
        # Mel spectrogram loss
        mel_loss = self.l1_loss(mel_output, mel_target) * mel_mask
        mel_loss = mel_loss.sum() / mel_mask.sum()
        
        # PostNet loss
        postnet_output = postnet_output[:, :min_t, :]
        postnet_loss = self.l1_loss(postnet_output, mel_target) * mel_mask
        postnet_loss = postnet_loss.sum() / mel_mask.sum()
        
        # Waveform loss - simplified approach
        # Just use MSE on the first min_len samples
        min_len = min(waveform.size(2), waveform_target.size(1))
        waveform_loss = self.mse_loss(
            waveform.squeeze(1)[:, :min_len], 
            waveform_target[:, :min_len]
        ).mean()
        
        # Total loss
        total_loss = (
            self.mel_weight * mel_loss +
            self.postnet_weight * postnet_loss + 
            self.waveform_weight * waveform_loss
        )
        
        # Return losses
        return total_loss, {
            'mel_loss': mel_loss.item(),
            'postnet_loss': postnet_loss.item(),
            'waveform_loss': waveform_loss.item(),
            'total_loss': total_loss.item()
        }


def get_dataloaders(metadata_path, batch_size, max_samples=None, num_workers=4):
    """
    Get dataloaders for training
    """
    # Create dataset
    dataset = LibriSpeechDataset(metadata_path, max_samples)
    
    # Split dataset
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def train(args):
    """
    Train the TTS model
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'samples'), exist_ok=True)
    
    # Initialize tensorboard
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))
    
    # Get dataloaders
    train_loader, val_loader = get_dataloaders(
        args.metadata_path,
        args.batch_size,
        args.max_samples,
        args.num_workers
    )
    print(f"Loaded dataloaders with {len(train_loader)} training batches and {len(val_loader)} validation batches")
    
    # Create model - use a smaller model for faster training
    base_model = TextToSpeechModel(
        vocab_size=args.vocab_size,
        embedding_dim=args.embedding_dim,
        encoder_layers=args.encoder_layers,
        encoder_heads=args.encoder_heads,
        encoder_dim=args.encoder_dim,
        duration_hidden=args.duration_predictor_filters,
        decoder_dim=args.decoder_dim,
        decoder_layers=args.decoder_layers,
        decoder_heads=args.decoder_heads,
        decoder_ffn_dim=args.decoder_ffn_dim,
        postnet_dim=args.postnet_dim,
        n_mels=args.n_mels,
        sample_rate=args.sample_rate,
        dropout=args.dropout
    )
    
    # Wrap the model with the adapter
    model = ModelAdapter(base_model)
    model = model.to(device)
    
    # Print model size
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params / 1e6:.2f} million trainable parameters")
    
    # Create loss function
    criterion = TTSLoss(
        mel_weight=args.mel_weight,
        duration_weight=args.duration_weight,
        postnet_weight=args.postnet_weight,
        waveform_weight=args.waveform_weight
    )
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=args.weight_decay
    )
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos',
        final_div_factor=args.lr_decay
    )
    
    # Training loop
    step = 0
    start_time = time.time()
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in progress_bar:
            # Move data to device
            text_indices = batch['text_indices'].to(device)
            mels = batch['mels'].to(device)
            waveforms = batch['waveforms'].to(device)
            text_lengths = batch['text_lengths'].to(device)
            mel_lengths = batch['mel_lengths'].to(device)
            
            # Forward pass
            outputs = model(text_indices)
            
            # Create targets dictionary
            targets = {
                'mel_spectrograms': mels,
                'waveforms': waveforms,
                'spectrogram_lengths': mel_lengths
            }
            
            # Calculate loss
            loss, losses_dict = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_thresh)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            
            # Update progress bar
            progress_bar.set_postfix(loss=loss.item())
            epoch_loss += loss.item()
            
            # Log to tensorboard
            writer.add_scalar('Loss/train', loss.item(), step)
            writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], step)
            for loss_name, loss_value in losses_dict.items():
                writer.add_scalar(f'Loss/{loss_name}', loss_value, step)
            
            step += 1
        
        # Calculate average epoch loss
        epoch_loss /= len(train_loader)
        print(f"Epoch {epoch+1} average loss: {epoch_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, 'checkpoints', f'model_epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': base_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': epoch_loss
        }, checkpoint_path)
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Validation"):
                    # Move data to device
                    text_indices = batch['text_indices'].to(device)
                    mels = batch['mels'].to(device)
                    waveforms = batch['waveforms'].to(device)
                    text_lengths = batch['text_lengths'].to(device)
                    mel_lengths = batch['mel_lengths'].to(device)
                    
                    # Forward pass
                    outputs = model(text_indices)
                    
                    # Create targets dictionary
                    targets = {
                        'mel_spectrograms': mels,
                        'waveforms': waveforms,
                        'spectrogram_lengths': mel_lengths
                    }
                    
                    # Calculate loss
                    loss, _ = criterion(outputs, targets)
                    val_loss += loss.item()
            
            # Calculate average validation loss
            val_loss /= len(val_loader)
            print(f"Validation loss: {val_loss:.4f}")
            
            # Log to tensorboard
            writer.add_scalar('Loss/validation', val_loss, epoch)
            
            # Generate sample every sample_interval epochs
            if (epoch + 1) % args.sample_interval == 0:
                # Generate sample from a validation example
                val_batch = next(iter(val_loader))
                text_indices = val_batch['text_indices'][0].unsqueeze(0).to(device)
                text_length = val_batch['text_lengths'][0].unsqueeze(0).to(device)
                text = val_batch['texts'][0]
                
                with torch.no_grad():
                    model.eval()
                    outputs = model(text_indices, text_length)
                    mel = outputs['postnet_output'][0]
                    waveform = outputs['waveform'][0, 0]
                
                # Save audio
                sample_path = os.path.join(args.output_dir, 'samples', f'sample_epoch_{epoch+1}.wav')
                waveform_np = waveform.cpu().numpy()
                import soundfile as sf
                sf.write(sample_path, waveform_np, args.sample_rate)
                
                # Log to tensorboard
                writer.add_audio(f'Generated Audio (Epoch {epoch+1})', waveform_np, 0, sample_rate=args.sample_rate)
                
                # Plot mel spectrogram
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(mel.cpu().numpy(), aspect='auto', origin='lower')
                ax.set_title(f'Generated Mel Spectrogram (Epoch {epoch+1})')
                ax.set_xlabel('Frame')
                ax.set_ylabel('Mel Channel')
                writer.add_figure('Mel Spectrogram', fig, epoch)
                plt.close(fig)
                
                # Log the text
                with open(os.path.join(args.output_dir, 'samples', f'sample_text_epoch_{epoch+1}.txt'), 'w') as f:
                    f.write(text)
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'model.pt')
    torch.save(base_model.state_dict(), final_model_path)
    
    # Final message
    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.2f} minutes")
    print(f"Final model saved to {final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TTS model on LibriSpeech")
    
    # Data parameters
    parser.add_argument("--metadata_path", type=str, default="preprocessed/metadata.json",
                        help="Path to metadata file")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save outputs")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to use")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    parser.add_argument("--lr_decay", type=float, default=50, help="Learning rate decay factor")
    parser.add_argument("--grad_clip_thresh", type=float, default=1.0, help="Gradient clipping threshold")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for data loading")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sample_interval", type=int, default=1, help="Generate sample every N epochs")
    
    # Loss weights
    parser.add_argument("--mel_weight", type=float, default=1.0, help="Weight for mel spectrogram loss")
    parser.add_argument("--duration_weight", type=float, default=0.1, help="Weight for duration loss")
    parser.add_argument("--postnet_weight", type=float, default=1.0, help="Weight for postnet loss")
    parser.add_argument("--waveform_weight", type=float, default=0.5, help="Weight for waveform loss")
    
    # Model parameters - use smaller values for faster training
    parser.add_argument("--vocab_size", type=int, default=100, help="Vocabulary size")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--encoder_layers", type=int, default=2, help="Number of encoder layers")
    parser.add_argument("--encoder_heads", type=int, default=2, help="Number of encoder attention heads")
    parser.add_argument("--encoder_dim", type=int, default=128, help="Encoder hidden dimension")
    parser.add_argument("--duration_predictor_filters", type=int, default=128, help="Duration predictor hidden dimension")
    parser.add_argument("--decoder_layers", type=int, default=2, help="Number of decoder layers")
    parser.add_argument("--decoder_dim", type=int, default=128, help="Decoder hidden dimension")
    parser.add_argument("--decoder_heads", type=int, default=2, help="Number of decoder attention heads")
    parser.add_argument("--decoder_ffn_dim", type=int, default=512, help="Decoder FFN dimension")
    parser.add_argument("--postnet_dim", type=int, default=128, help="PostNet dimension")
    parser.add_argument("--n_mels", type=int, default=80, help="Number of mel spectrogram bins")
    parser.add_argument("--sample_rate", type=int, default=22050, help="Audio sample rate")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    train(args) 