"""
Example usage of NISQA inference module
"""

import numpy as np
from nisqa import NisqaModel

# Initialize model (uses default weights/nisqa.tar)
model = NisqaModel()

# Example 1: Predict from file path
result = model(filepath='path/to/audio.wav')
print("Predictions from file:")
print(f"  MOS:     {result['mos_pred']:.2f}")
print(f"  Noisiness:   {result['noi_pred']:.2f}")
print(f"  Coloration:  {result['col_pred']:.2f}")
print(f"  Discontinuity: {result['dis_pred']:.2f}")
print(f"  Loudness:    {result['loud_pred']:.2f}")

# Example 2: Predict from waveform (numpy array)
# Assuming you have loaded audio as a numpy array
# audio_array = librosa.load('path/to/audio.wav', sr=None)[0]
# result = model(waveform=audio_array)
# print("\nPredictions from waveform:")
# print(result)

# Example 3: Using different model weights
# model_tts = NisqaModel(model_path='weights/nisqa_tts.tar')
# result_tts = model_tts(filepath='path/to/synthesized_speech.wav')
# print(f"Naturalness: {result_tts['mos_pred']:.2f}")

# Example 4: Specify device explicitly
# model_cpu = NisqaModel(device='cpu')
# model_gpu = NisqaModel(device='cuda')
