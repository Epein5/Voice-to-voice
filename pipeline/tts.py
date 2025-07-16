"""
TTSHandler: Simple Text-to-Speech
"""
import torch
from transformers import VitsModel, AutoTokenizer
import numpy as np
import soundfile as sf
import tempfile

class TTSHandler:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load model and tokenizer ONCE at startup
        self.model = VitsModel.from_pretrained("procit001/nepali_male_v1").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("procit001/nepali_male_v1")

    def synthesize(self, text: str) -> bytes:
        # Only inference here, no reloading
        print(f"Synthesizing text: {text}")
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(**inputs).waveform
        audio_data = output.cpu().numpy().flatten()
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_data, self.model.config.sampling_rate, format='WAV', subtype='PCM_16')
            with open(tmp_file.name, 'rb') as f:
                audio_bytes = f.read()
        return audio_bytes
