# Text-to-Speech Model with LibriSpeech

This project implements a neural text-to-speech (TTS) model and fine-tunes it on the LibriSpeech dev-clean dataset.

## Architecture

The model uses a Transformer-based architecture with:
- Text encoder that converts text input to phonemes and then to embeddings
- Duration predictor that estimates how long each phoneme should be pronounced
- Mel-spectrogram decoder that generates audio features
- Vocoder that converts spectrograms to audio waveforms

## Dataset

The model is fine-tuned using the LibriSpeech dev-clean dataset, which contains high-quality English speech recordings.

## Usage

1. Install dependencies:
```
pip install -r requirements.txt
```

2. Preprocess the LibriSpeech dataset:
```
python preprocess.py
```

3. Train the model:
```
python train.py
```

4. Generate audio from text:
```
python generate.py --text "Text to synthesize" --output output.wav
```

## Requirements

See `requirements.txt` for the list of dependencies. 