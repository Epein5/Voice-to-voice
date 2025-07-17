import numpy as np
import logging
from typing import Optional, Tuple
import io

logger = logging.getLogger(__name__)

class VADHandler:
    """Voice Activity Detection handler using energy-based detection"""
    
    def __init__(self, 
                 energy_threshold: float = 0.01,
                 silence_duration: float = 1.5,
                 min_speech_duration: float = 0.5,
                 sample_rate: int = 16000):
        """
        Initialize VAD handler
        
        Args:
            energy_threshold: Minimum energy level to consider as speech
            silence_duration: Duration of silence (in seconds) to consider speech ended
            min_speech_duration: Minimum duration (in seconds) for valid speech
            sample_rate: Audio sample rate
        """
        self.energy_threshold = energy_threshold
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration
        self.sample_rate = sample_rate
        
        # State tracking
        self.is_speaking = False
        self.speech_start_time = None
        self.last_speech_time = None
        self.audio_buffer = []
        
    def calculate_energy(self, audio_chunk: np.ndarray) -> float:
        """Calculate the energy (RMS) of an audio chunk"""
        if len(audio_chunk) == 0:
            return 0.0
        return np.sqrt(np.mean(audio_chunk ** 2))
    
    def process_audio_chunk(self, audio_chunk: np.ndarray, timestamp: float) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Process an audio chunk and determine if speech is detected
        
        Args:
            audio_chunk: Audio data as numpy array
            timestamp: Current timestamp in seconds
            
        Returns:
            Tuple of (speech_detected, complete_speech_segment)
            speech_detected: True if currently detecting speech
            complete_speech_segment: Audio data if a complete speech segment is ready
        """
        energy = self.calculate_energy(audio_chunk)
        
        # Check if current chunk has speech
        has_speech = energy > self.energy_threshold
        
        if has_speech:
            if not self.is_speaking:
                # Start of speech detected
                self.is_speaking = True
                self.speech_start_time = timestamp
                self.audio_buffer = [audio_chunk]
                logger.debug(f"Speech started at {timestamp:.2f}s, energy: {energy:.4f}")
            else:
                # Continue speech
                self.audio_buffer.append(audio_chunk)
            
            self.last_speech_time = timestamp
            
        else:
            if self.is_speaking:
                # Check if silence duration exceeded
                silence_duration = timestamp - self.last_speech_time
                
                if silence_duration >= self.silence_duration:
                    # End of speech detected
                    speech_duration = timestamp - self.speech_start_time
                    
                    if speech_duration >= self.min_speech_duration:
                        # Valid speech segment
                        complete_audio = np.concatenate(self.audio_buffer)
                        logger.info(f"Speech ended. Duration: {speech_duration:.2f}s, "
                                  f"Samples: {len(complete_audio)}")
                        
                        # Reset state
                        self.is_speaking = False
                        self.speech_start_time = None
                        self.last_speech_time = None
                        self.audio_buffer = []
                        
                        return False, complete_audio
                    else:
                        # Too short, discard
                        logger.debug(f"Speech too short ({speech_duration:.2f}s), discarding")
                        self.is_speaking = False
                        self.speech_start_time = None
                        self.last_speech_time = None
                        self.audio_buffer = []
                else:
                    # Still in potential speech, add silence to buffer
                    self.audio_buffer.append(audio_chunk)
        
        return self.is_speaking, None
    
    def reset(self):
        """Reset the VAD state"""
        self.is_speaking = False
        self.speech_start_time = None
        self.last_speech_time = None
        self.audio_buffer = []
        
    def set_sensitivity(self, sensitivity: float):
        """
        Adjust sensitivity (0.0 to 1.0)
        Higher values = more sensitive (detects quieter sounds)
        Lower values = less sensitive (only loud sounds)
        """
        # Map sensitivity to energy threshold - adjusted for louder speech requirement
        # sensitivity 0.0 -> threshold 0.2 (much less sensitive, need loud speech)
        # sensitivity 1.0 -> threshold 0.01 (more sensitive)
        self.energy_threshold = 0.2 * (1.0 - sensitivity) + 0.01 * sensitivity
        logger.info(f"VAD sensitivity set to {sensitivity:.2f}, threshold: {self.energy_threshold:.4f}")


class AudioProcessor:
    """Helper class for audio processing and chunking"""
    
    def __init__(self, chunk_duration: float = 0.1, sample_rate: int = 16000):
        """
        Initialize audio processor
        
        Args:
            chunk_duration: Duration of each audio chunk in seconds
            sample_rate: Audio sample rate
        """
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.chunk_size = int(chunk_duration * sample_rate)
        self.buffer = np.array([])
        
    def add_audio(self, audio_data: np.ndarray) -> list:
        """
        Add audio data and return complete chunks
        
        Args:
            audio_data: New audio data to add
            
        Returns:
            List of complete audio chunks
        """
        # Add new data to buffer
        self.buffer = np.concatenate([self.buffer, audio_data])
        
        # Extract complete chunks
        chunks = []
        while len(self.buffer) >= self.chunk_size:
            chunk = self.buffer[:self.chunk_size]
            chunks.append(chunk)
            self.buffer = self.buffer[self.chunk_size:]
            
        return chunks
    
    def get_remaining(self) -> Optional[np.ndarray]:
        """Get any remaining audio data in buffer"""
        if len(self.buffer) > 0:
            remaining = self.buffer.copy()
            self.buffer = np.array([])
            return remaining
        return None
