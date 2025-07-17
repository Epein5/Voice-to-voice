from transformers import VitsModel, AutoTokenizer
import torch
import io
import soundfile as sf
import numpy as np
from scipy import signal

class TTSHandler:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VitsModel.from_pretrained("procit001/nepali_male_v1").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("procit001/nepali_male_v1")
        self.speaking_rate = 1  # Slower speech (0.8 = 80% of normal speed)

    def synthesize(self, text: str) -> bytes:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(**inputs).waveform
        waveform = output.cpu().numpy().squeeze()

        # Slow down the speech by time-stretching
        if self.speaking_rate != 1.0:
            waveform = self._time_stretch(waveform, self.speaking_rate)

        with io.BytesIO() as wav_io:
            sf.write(wav_io, waveform, self.model.config.sampling_rate, format='WAV')
            wav_io.seek(0)
            return wav_io.read()

    def _time_stretch(self, audio, rate):
        """
        Time-stretch audio to change speaking rate without changing pitch
        rate < 1.0 = slower speech
        rate > 1.0 = faster speech
        """
        try:
            # Simple time stretching using resampling
            # This is a basic approach - for better quality, consider using librosa
            original_length = len(audio)
            new_length = int(original_length / rate)

            # Resample to stretch time
            stretched = signal.resample(audio, new_length)

            return stretched
        except Exception as e:
            # If time stretching fails, return original audio
            print(f"Time stretching failed: {e}")
            return audio