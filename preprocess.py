import os
import argparse
import json
import torch
import torchaudio
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def process_audio_file(args):
    """
    Process a single audio file
    """
    audio_file, output_dir, sample_rate, n_mels, text = args
    
    # Create file ID
    file_id = os.path.basename(audio_file).split('.')[0]
    
    try:
        # Load audio file
        waveform, orig_sr = torchaudio.load(audio_file)
        waveform = waveform.mean(dim=0)  # Convert to mono
        
        # Resample if needed
        if orig_sr != sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sr, sample_rate)
            waveform = resampler(waveform)
        
        # Compute mel spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=n_mels
        )
        
        mel_spec = mel_transform(waveform)
        log_mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
        
        # Skip phonemization since we don't have eSpeak
        # Just use the text as-is
        phonemes = text.lower()
        
        # Create output paths
        waveform_path = os.path.join(output_dir, 'waveforms', f"{file_id}.pt")
        mel_path = os.path.join(output_dir, 'mels', f"{file_id}.pt")
        
        # Save tensors
        torch.save(waveform, waveform_path)
        torch.save(log_mel_spec, mel_path)
        
        # Return metadata
        return {
            'file_id': file_id,
            'text': text,
            'phonemes': phonemes,
            'waveform_path': waveform_path,
            'mel_path': mel_path,
            'waveform_length': waveform.size(0),
            'spec_length': log_mel_spec.size(1)
        }
    
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None


def main(args):
    """
    Preprocess LibriSpeech dataset
    """
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'waveforms'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'mels'), exist_ok=True)
    
    # Collect audio files and transcriptions
    files = []
    transcripts = {}
    
    print("Collecting files...")
    for speaker_id in tqdm(os.listdir(args.data_dir)):
        speaker_dir = os.path.join(args.data_dir, speaker_id)
        if not os.path.isdir(speaker_dir):
            continue
        
        for chapter_id in os.listdir(speaker_dir):
            chapter_dir = os.path.join(speaker_dir, chapter_id)
            if not os.path.isdir(chapter_dir):
                continue
            
            # Load transcription file
            trans_file = os.path.join(chapter_dir, f"{speaker_id}-{chapter_id}.trans.txt")
            if os.path.exists(trans_file):
                with open(trans_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split(' ', 1)
                        if len(parts) == 2:
                            file_id, text = parts
                            transcripts[file_id] = text
                            
                            # Check if audio file exists
                            audio_file = os.path.join(chapter_dir, f"{file_id}.flac")
                            if os.path.exists(audio_file):
                                files.append((audio_file, args.output_dir, args.sample_rate, args.n_mels, text))
    
    print(f"Found {len(files)} audio files")
    
    # Process a subset if specified
    if args.max_files > 0:
        files = files[:args.max_files]
        print(f"Processing {len(files)} files as specified by max_files")
    
    # Process files in parallel
    metadata = []
    with Pool(processes=args.num_workers) as pool:
        for result in tqdm(pool.imap(process_audio_file, files), total=len(files)):
            if result:
                metadata.append(result)
    
    # Save metadata
    metadata_path = os.path.join(args.output_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Preprocessing complete. Processed {len(metadata)} files.")
    print(f"Metadata saved to {metadata_path}")
    
    # Only generate stats if we have processed files
    if metadata:
        # Generate stats
        print("\nGenerating dataset statistics...")
        waveform_lengths = [item['waveform_length'] for item in metadata]
        spec_lengths = [item['spec_length'] for item in metadata]
        
        print(f"Total audio duration: {sum(waveform_lengths) / args.sample_rate / 3600:.2f} hours")
        print(f"Average audio duration: {np.mean(waveform_lengths) / args.sample_rate:.2f} seconds")
        print(f"Min audio duration: {min(waveform_lengths) / args.sample_rate:.2f} seconds")
        print(f"Max audio duration: {max(waveform_lengths) / args.sample_rate:.2f} seconds")
        
        # Plot histogram of audio durations
        plt.figure(figsize=(10, 4))
        plt.hist([length / args.sample_rate for length in waveform_lengths], bins=50)
        plt.xlabel('Duration (seconds)')
        plt.ylabel('Count')
        plt.title('Audio Duration Distribution')
        plt.savefig(os.path.join(args.output_dir, 'duration_histogram.png'))
        plt.close()
        
        # Plot histogram of spectrogram lengths
        plt.figure(figsize=(10, 4))
        plt.hist(spec_lengths, bins=50)
        plt.xlabel('Spectrogram Length (frames)')
        plt.ylabel('Count')
        plt.title('Spectrogram Length Distribution')
        plt.savefig(os.path.join(args.output_dir, 'spec_length_histogram.png'))
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess LibriSpeech dataset")
    
    parser.add_argument("--data_dir", type=str, default="data/LibriSpeech/dev-clean",
                        help="Path to LibriSpeech directory")
    parser.add_argument("--output_dir", type=str, default="preprocessed",
                        help="Directory to save preprocessed data")
    parser.add_argument("--sample_rate", type=int, default=22050,
                        help="Target sample rate")
    parser.add_argument("--n_mels", type=int, default=80,
                        help="Number of mel spectrogram bins")
    parser.add_argument("--num_workers", type=int, default=cpu_count(),
                        help="Number of workers for parallel processing")
    parser.add_argument("--max_files", type=int, default=100,
                        help="Maximum number of files to process (0 for all)")
    
    args = parser.parse_args()
    
    main(args) 