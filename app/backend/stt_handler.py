from transformers import pipeline
import io
import soundfile as sf
import tempfile
import os
import logging

logger = logging.getLogger(__name__)

class STTHandler:
    def __init__(self):
        # Load the Nepali ASR model
        self.pipe = pipeline("automatic-speech-recognition", model="amitpant7/whispher-nepali-asr")

    def transcribe(self, audio_bytes: bytes) -> str:
        try:
            # Create a temporary file to handle the audio data
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_file_path = temp_file.name

            try:
                # Try to read the audio file directly
                data, samplerate = sf.read(temp_file_path)

                # Create a WAV file in memory for the pipeline
                with io.BytesIO() as wav_buffer:
                    sf.write(wav_buffer, data, samplerate, format='WAV')
                    wav_buffer.seek(0)

                    # Use the pipeline with the WAV data
                    result = self.pipe(wav_buffer.read())

                return result['text']

            finally:
                # Clean up the temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            # Fallback: try to process as raw audio data
            try:
                # Try different approaches for audio processing
                with io.BytesIO(audio_bytes) as audio_buffer:
                    # Try to use the pipeline directly with bytes
                    result = self.pipe(audio_buffer.read())
                    return result['text']
            except Exception as e2:
                logger.error(f"Fallback transcription also failed: {e2}")
                raise Exception(f"Could not process audio: {e}")