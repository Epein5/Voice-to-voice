"""
STTHandler: Simple Speech-to-Text
"""
from transformers import pipeline
import tempfile

class STTHandler:
    def __init__(self):
        self.model = pipeline("automatic-speech-recognition", model="amitpant7/whispher-nepali-asr")

    def transcribe(self, audio_data: bytes) -> str:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_file.write(audio_data)
            tmp_file_path = tmp_file.name
        result = self.model(tmp_file_path)
        print(f"Transcription result: {result}")
        return result['text']
