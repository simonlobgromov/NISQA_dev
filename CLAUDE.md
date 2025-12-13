# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

NISQA (Non-Intrusive Speech Quality and Naturalness Assessment) is an **inference-only** deep learning module for predicting speech quality. This is a refactored version that removed all training and evaluation code to provide a clean, simple API.

**Key Features:**
- **Inference Only**: No training code - just load model and predict
- **Simple API**: Single class (`NisqaModel`) with `__call__` method
- **Pip-based**: Uses `requirements.txt` instead of conda
- **Dual Input**: Accepts both file paths and waveform arrays

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: torch, numpy, scipy, librosa, soundfile, pyyaml, pandas, tqdm

## Project Structure

```
NISQA_dev/
├── nisqa/
│   ├── __init__.py          # Package init, exports NisqaModel
│   ├── inference.py         # Main inference module (NEW - contains everything)
│   ├── NISQA_lib.py.backup  # Old original library (backup)
│   └── NISQA_model.py.backup # Old model class (backup)
├── weights/
│   ├── nisqa.tar            # NISQA v2.0 multidimensional model
│   ├── nisqa_mos_only.tar   # MOS-only model
│   └── nisqa_tts.tar        # TTS naturalness model
├── example.py               # Usage examples
├── requirements.txt         # Pip dependencies
├── README.md               # User documentation
└── CLAUDE.md               # This file

Removed files (training-related):
- run_train.py
- run_evaluate.py
- run_predict.py
- config/ directory
- env.yml
```

## Main Module: nisqa/inference.py

This single file contains everything needed for inference:

### Neural Network Classes
- `NISQA_DIM`: Main model (MOS + 4 dimensions: Noisiness, Coloration, Discontinuity, Loudness)
- `Framewise`: CNN-based framewise feature extraction
- `TimeDependency`: Self-Attention or LSTM for temporal modeling
- `Pooling`: Attention pooling to aggregate features

### Preprocessing Functions
- `get_librosa_melspec()`: Load audio file → mel-spectrogram
- `waveform_to_melspec()`: Convert waveform array → mel-spectrogram
- `segment_specs()`: Segment mel-spec into CNN-friendly chunks

### Main API Class
```python
class NisqaModel:
    def __init__(self, model_path='weights/nisqa.tar', device=None)
    def __call__(self, waveform=None, filepath=None) -> dict
```

## Usage

### Basic Usage

```python
from nisqa import NisqaModel

# Initialize (loads weights/nisqa.tar by default)
model = NisqaModel()

# Predict from file
result = model(filepath='audio.wav')

# Predict from waveform (numpy or torch tensor)
import numpy as np
audio = np.random.randn(48000)
result = model(waveform=audio)
```

### Output Format

Returns dict with 5 keys:
```python
{
    'mos_pred': float,   # Overall quality (1-5)
    'noi_pred': float,   # Noisiness (1-5)
    'dis_pred': float,   # Discontinuity (1-5)
    'col_pred': float,   # Coloration (1-5)
    'loud_pred': float   # Loudness (1-5)
}
```

### Different Models

```python
# Default: NISQA v2.0 multidimensional
model = NisqaModel()

# MOS only (for finetuning in original repo)
model = NisqaModel(model_path='weights/nisqa_mos_only.tar')

# TTS naturalness prediction
model_tts = NisqaModel(model_path='weights/nisqa_tts.tar')
```

## Model Checkpoint Format

The `.tar` files in `weights/` contain:
- `model_state_dict`: PyTorch state dict
- `args`: Dictionary with model architecture parameters
- `model_args`: Subset of args for model construction
- Additional training metadata (not used in inference)

The `NisqaModel.__init__` method:
1. Loads checkpoint with `torch.load()`
2. Extracts model args from `checkpoint['args']`
3. Constructs `NISQA_DIM` model with those args
4. Loads weights from `checkpoint['model_state_dict']`
5. Sets model to eval mode

## Audio Processing

- Default sample rate: 48kHz (auto-resampled by librosa)
- Mel-spectrogram: 48 bands, 20kHz max frequency
- Window: 40ms, Hop: 10ms
- Segments: 15 bins width
- Max duration: ~52 seconds (1300 segments)

Both mono and stereo files supported (stereo converted to mono mix).

## Common Development Tasks

### Testing the module
```bash
python example.py
```

### Adding new model support
1. Place `.tar` file in `weights/`
2. Initialize with `NisqaModel(model_path='weights/your_model.tar')`
3. Ensure checkpoint has compatible keys

### Debugging
Check that:
- PyTorch can load the checkpoint
- Model architecture matches checkpoint args
- Input audio is valid (not too short, readable format)

## Important Notes

- **Inference only**: No training capabilities in this version
- **Device auto-detection**: Uses CUDA if available, otherwise CPU
- **Batch size**: Handles single samples (batch dimension added/removed automatically)
- **No CLI**: All interaction through Python API
- **Backup files**: Original `NISQA_lib.py` and `NISQA_model.py` preserved as `.backup` files
