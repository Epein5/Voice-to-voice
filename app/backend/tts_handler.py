from transformers import VitsModel, AutoTokenizer
import torch
import io
import soundfile as sf

class TTSHandler:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VitsModel.from_pretrained("procit001/nepali_male_v1").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("procit001/nepali_male_v1")

    def synthesize(self, text: str) -> bytes:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(**inputs).waveform
        waveform = output.cpu().numpy().squeeze()
        with io.BytesIO() as wav_io:
            sf.write(wav_io, waveform, self.model.config.sampling_rate, format='WAV')
            wav_io.seek(0)
            return wav_io.read() 