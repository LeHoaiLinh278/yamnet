import librosa
import soundfile as sf
import numpy as np
import resampy
import params
def read_wav(fname, output_sr=16000, use_rosa=False):
    # small wrapper - i was seeing some slightly different 
    # results when loading with different libraries 
    if use_rosa:
        waveform, sr = librosa.load(fname, sr=output_sr)
    else:
        wav_data, sr = sf.read(fname, dtype=np.int16)
        assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
        waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]
        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=1)
        if sr != params.SAMPLE_RATE:
            waveform = resampy.resample(waveform, sr, params.SAMPLE_RATE)
    
    return waveform, sr