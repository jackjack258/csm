import os
import sys
import time
import torch
import torchaudio
import tempfile
import numpy as np
import json
import wave
import logging
import random
import threading
from pathlib import Path
from typing import List, Optional, Dict, Tuple

# Filter out PyQt5 deprecation warnings
import warnings
warnings.filterwarnings("ignore", message="sipPyTypeDict.*")

# Set CUDA environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# UI imports
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                            QHBoxLayout, QWidget, QTextEdit, QLabel, QFileDialog, 
                            QProgressBar, QMessageBox, QComboBox, QSlider, 
                            QLineEdit, QGridLayout, QInputDialog, QDialog)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer, QUrl, QObject
from PyQt5.QtGui import QPalette, QColor, QFont, QPainter
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

# Audio imports
import pyaudio
import wave

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SILENCE_THRESHOLD = 700
MIN_SILENCE_DURATION = 1.0
ASSISTANT_VOICE_DIR = "assistant_voice_samples"
USER_VOICE_DIR = "user_voice_samples"

# Random user utterances for sample context
USER_UTTERANCES = [
    "Hey, how are you doing today?",
    "What's the weather like?",
    "Can you tell me more about that?",
    "That's interesting, tell me more.",
    "I have a question about something.",
    "Could you explain that in more detail?",
    "What do you think about this?",
    "I'm not sure I understand.",
    "That makes sense, thanks.",
    "I appreciate your help with this.",
    "Can you give me some advice?",
    "What would you recommend?",
    "I'm curious about your opinion.",
    "Let's talk about something else.",
    "I wonder if you could help me with something."
]

def check_system_resources():
    """Check available system resources and provide warnings if needed"""
    try:
        import psutil
        
        # Check available RAM
        available_ram_gb = psutil.virtual_memory().available / (1024**3)
        logger.info(f"Available RAM: {available_ram_gb:.2f} GB")
        
        # Check CUDA memory if available
        cuda_info = ""
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            total_cuda_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            free_cuda_mem = torch.cuda.memory_reserved(device) / (1024**3)
            cuda_info = f", CUDA memory: {free_cuda_mem:.2f} GB free of {total_cuda_mem:.2f} GB"
            logger.info(f"CUDA device: {torch.cuda.get_device_name(device)}{cuda_info}")
        
        if available_ram_gb < 4:
            logger.warning(f"Low memory warning: Only {available_ram_gb:.2f} GB RAM available")
            return False, f"Low memory warning: Only {available_ram_gb:.2f} GB RAM available{cuda_info}"
        
        return True, f"System check passed: {available_ram_gb:.2f} GB RAM available{cuda_info}"
    except Exception as e:
        logger.error(f"Error checking system resources: {e}")
        return True, "Could not check system resources"


class VoiceActivityDetector:
    """Detects when user stops speaking"""
    
    def __init__(self, threshold=SILENCE_THRESHOLD, min_silence_duration=MIN_SILENCE_DURATION):
        self.threshold = threshold
        self.min_silence_duration = min_silence_duration
        self.silence_start_time = None
        self.is_silence = False
        
    def process_audio(self, audio_data):
        """Process audio chunk to detect silence"""
        if isinstance(audio_data, bytes):
            audio_data = np.frombuffer(audio_data, dtype=np.int16)
        
        # Calculate audio level (RMS)
        audio_level = np.sqrt(np.mean(np.square(audio_data)))
        
        # Detect if current chunk is silence
        current_is_silence = audio_level < self.threshold
        
        # State transitions
        if not self.is_silence and current_is_silence:
            # Transition to silence
            self.silence_start_time = time.time()
            self.is_silence = True
            return False
        elif self.is_silence and not current_is_silence:
            # Transition back to speech
            self.is_silence = False
            self.silence_start_time = None
            return False
        elif self.is_silence and current_is_silence:
            # Continued silence - check if it's been long enough
            if time.time() - self.silence_start_time >= self.min_silence_duration:
                # Speech ended
                self.is_silence = False
                self.silence_start_time = None
                return True
        
        # No transition or not enough silence
        return False
    
    def reset(self):
        """Reset the detector state"""
        self.silence_start_time = None
        self.is_silence = False


class VoiceWaveformWidget(QWidget):
    """Simple widget to display audio waveform/levels"""
    
    def __init__(self, parent=None):
        super().__init__()
        self.setMinimumHeight(50)
        self.setMinimumWidth(200)
        
        # Audio level history (for waveform)
        self.audio_levels = []
        self.max_history = 100  # Number of points to keep
        self.threshold = SILENCE_THRESHOLD  # Threshold line
        
    def set_threshold(self, threshold):
        self.threshold = threshold
        self.update()
        
    def update_level(self, level):
        # Add new level and maintain history size
        self.audio_levels.append(min(level, 10000))
        if len(self.audio_levels) > self.max_history:
            self.audio_levels.pop(0)
        self.update()
        
    def clear_levels(self):
        self.audio_levels = []
        self.update()
        
    def paintEvent(self, event):
        if not self.audio_levels:
            return
            
        from PyQt5.QtGui import QPainter, QColor
        
        # Setup painting
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        width = self.width()
        height = self.height()
        
        # Background
        painter.fillRect(0, 0, width, height, QColor(20, 20, 20))
        
        # Draw threshold line - convert float to int to fix TypeError
        painter.setPen(QColor(255, 150, 0))
        threshold_y = int(height - (height * self.threshold / 10000))
        painter.drawLine(0, threshold_y, width, threshold_y)
        
        # Draw waveform
        painter.setPen(QColor(40, 210, 40))
        
        # Scale x based on number of points
        x_scale = width / max(1, self.max_history - 1)
        
        for i in range(1, len(self.audio_levels)):
            # Scale values to fit height and invert (0 at bottom)
            # Convert float to int to fix TypeError
            y1 = int(height - (height * self.audio_levels[i-1] / 10000))
            y2 = int(height - (height * self.audio_levels[i] / 10000))
            
            x1 = int((i-1) * x_scale)
            x2 = int(i * x_scale)
            
            painter.drawLine(x1, y1, x2, y2)


class AudioRecorder(QThread):
    """Records audio and handles automatic detection of speech end"""
    
    finished_signal = pyqtSignal(str)
    audio_level_signal = pyqtSignal(float)
    status_signal = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__()
        self.chunk = CHUNK_SIZE
        self.format = FORMAT
        self.channels = CHANNELS
        self.rate = RATE
        self.is_recording = False
        self.frames = []
        self.vad = VoiceActivityDetector()
        self.auto_stop = True  # Flag to control whether to auto-stop on silence
        self.output_file = None  # Path to save recording
        
    def run(self):
        try:
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(format=self.format,
                                      channels=self.channels,
                                      rate=self.rate,
                                      input=True,
                                      frames_per_buffer=self.chunk)
            
            self.frames = []
            self.is_recording = True
            self.vad.reset()
            
            self.status_signal.emit("Listening...")
            
            while self.is_recording:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                self.frames.append(data)
                
                # Calculate and emit audio level for visualization
                audio_data = np.frombuffer(data, dtype=np.int16)
                audio_level = np.sqrt(np.mean(np.square(audio_data)))
                self.audio_level_signal.emit(audio_level)
                
                # Check for end of speech if auto-stop is enabled
                if self.auto_stop and self.vad.process_audio(data):
                    self.status_signal.emit("Speech detected, stopping...")
                    self.is_recording = False
            
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
            
            # Only process if we have audio frames
            if len(self.frames) > 0:
                # Save recorded audio to a temporary file
                if not self.output_file:
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                    temp_file.close()
                    self.output_file = temp_file.name
                
                wf = wave.open(self.output_file, 'wb')
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.p.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(self.frames))
                wf.close()
                
                # Also save to the user voice samples directory for future use
                os.makedirs(USER_VOICE_DIR, exist_ok=True)
                user_sample_path = os.path.join(USER_VOICE_DIR, f"user_sample_{int(time.time())}.wav")
                
                # Copy the file to user samples
                with open(self.output_file, 'rb') as src_file:
                    with open(user_sample_path, 'wb') as dst_file:
                        dst_file.write(src_file.read())
                
                logger.info(f"Recorded audio saved to {self.output_file} and {user_sample_path}")
                self.finished_signal.emit(self.output_file)
            else:
                logger.warning("No audio frames recorded")
                self.finished_signal.emit("")
                
        except Exception as e:
            logger.error(f"Error in audio recording: {e}")
            self.status_signal.emit(f"Error: {str(e)}")
            self.finished_signal.emit("")
        
    def stop(self):
        """Stop recording"""
        if self.is_recording:
            self.is_recording = False
    
    def set_auto_stop(self, enabled):
        """Enable or disable automatic stop on silence"""
        self.auto_stop = enabled
    
    def set_output_file(self, file_path):
        """Set a specific output file path"""
        self.output_file = file_path


# Audio player for playback
class AudioPlayer(QObject):
    """Simple audio player using QMediaPlayer"""
    
    finished_signal = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.player = QMediaPlayer()
        self.player.setVolume(100)
        self.player.stateChanged.connect(self._on_state_changed)
    
    def play(self, file_path):
        """Play an audio file"""
        if not os.path.exists(file_path):
            logger.error(f"Audio file not found: {file_path}")
            return False
            
        try:
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(file_path)))
            self.player.play()
            return True
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
            return False
    
    def stop(self):
        """Stop playback"""
        self.player.stop()
    
    def _on_state_changed(self, state):
        """Handle player state changes"""
        if state == QMediaPlayer.StoppedState:
            self.finished_signal.emit()


# Speech recognition worker using Whisper
class SpeechRecognitionWorker(QThread):
    result_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    model_ready_signal = pyqtSignal(bool)
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.audio_file = None
        self.transcribe_mode = "single"  # "single" or "batch"
        self.batch_files = []
        self.results = {}
        
    def load_model(self):
        """Load Whisper model"""
        import whisper
        
        self.status_signal.emit("Loading Whisper model...")
        self.progress_signal.emit(10)
        
        # Load the model using whisper
        try:
            self.model = whisper.load_model("medium", device="cuda")
            
            self.progress_signal.emit(100)
            self.status_signal.emit("Whisper model loaded")
            logger.info("Whisper model loaded successfully")
            self.model_ready_signal.emit(True)
            return True
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            self.model_ready_signal.emit(False)
            return False
        
    def set_audio_file(self, file_path):
        """Set the path to the audio file for transcription"""
        self.audio_file = file_path
        self.transcribe_mode = "single"
        
    def set_batch_files(self, file_paths):
        """Set multiple files for batch transcription"""
        self.batch_files = file_paths
        self.transcribe_mode = "batch"
        self.results = {}
        
    def run(self):
        """Run transcription"""
        if not self.model:
            logger.error("Whisper model not loaded")
            return
            
        if self.transcribe_mode == "single":
            self._transcribe_single_file()
        else:
            self._transcribe_batch()
            
    def _transcribe_single_file(self):
        """Transcribe a single audio file"""
        if not self.audio_file:
            return
            
        try:
            self.status_signal.emit("Transcribing audio...")
            self.progress_signal.emit(10)
            
            # Run transcription using Whisper
            result = self.model.transcribe(
                self.audio_file,
                language="en"
            )
            
            # Extract the transcribed text
            text = result["text"].strip()
            
            # If transcription is very short, try to get more context
            if len(text) <= 3:  # Very short text like "it." or "hi"
                logger.info(f"Very short transcription: '{text}', trying again with more context")
                # Try again with different parameters
                result = self.model.transcribe(
                    self.audio_file,
                    language="en",
                    beam_size=5,
                    best_of=5
                )
                new_text = result["text"].strip()
                if len(new_text) > len(text):
                    text = new_text
            
            logger.info(f"Transcription result: '{text}'")
            
            # Provide a minimum fallback text for empty results
            if not text:
                text = "Hello"
                logger.warning(f"Empty transcription, using fallback: '{text}'")
            
            self.progress_signal.emit(100)
            self.status_signal.emit("Transcription complete")
            self.result_signal.emit(text)
            
        except Exception as e:
            logger.error(f"Error in speech recognition: {e}")
            self.result_signal.emit("Hello")  # Fallback text
            
    def _transcribe_batch(self):
        """Transcribe a batch of audio files"""
        if not self.batch_files:
            return
            
        try:
            total_files = len(self.batch_files)
            self.status_signal.emit(f"Transcribing {total_files} audio files...")
            
            for i, file_path in enumerate(self.batch_files):
                file_name = os.path.basename(file_path)
                self.status_signal.emit(f"Transcribing file {i+1}/{total_files}: {file_name}")
                self.progress_signal.emit(int((i / total_files) * 100))
                
                try:
                    # Run transcription using Whisper
                    result = self.model.transcribe(
                        file_path,
                        language="en"
                    )
                    
                    # Extract the transcribed text
                    text = result["text"].strip()
                    if not text:
                        text = f"Sample {i+1}"  # Fallback if no transcription
                        
                    self.results[file_name] = text
                    logger.info(f"Transcribed '{file_name}': '{text}'")
                except Exception as e:
                    logger.error(f"Error transcribing {file_name}: {e}")
                    self.results[file_name] = f"Sample {i+1}"  # Fallback
            
            self.progress_signal.emit(100)
            self.status_signal.emit(f"Transcribed {len(self.results)} files successfully")
            self.result_signal.emit(json.dumps(self.results))
            
        except Exception as e:
            logger.error(f"Error in batch transcription: {e}")
            self.result_signal.emit("")


# Text generation worker using LLM
class TextGenerationWorker(QThread):
    result_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.model_path = ""
        self.prompt = ""
        self.max_tokens = 2560
        self.temperature = 0.7
        self.conversation_history = []
        self.system_prompt = (
            "You are a helpful voice assistant. Respond in a natural, conversational manner. "
            "Keep your responses fairly concise but informative. Be friendly and helpful."
        )
        
    def set_model_path(self, model_path):
        self.model_path = model_path
        
    def set_system_prompt(self, system_prompt):
        self.system_prompt = system_prompt
        
    def load_model(self):
        try:
            self.status_signal.emit(f"Loading LLM model...")
            self.progress_signal.emit(10)
            
            # Import llama-cpp
            from llama_cpp import Llama
            
            # Calculate optimal n_gpu_layers based on model size and GPU memory
            n_gpu_layers = -1  # Default to all layers
            try:
                model_size_gb = os.path.getsize(self.model_path) / (1024**3)
                if torch.cuda.is_available():
                    device = torch.cuda.current_device()
                    cuda_mem_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
                    
                    # Reserve some GPU memory for other processes
                    available_mem = cuda_mem_gb * 0.9
                    
                    # Estimate how many layers we can fit
                    # Rough estimate: each 1B params uses ~2GB in f16
                    if model_size_gb > available_mem:
                        # Rough calculation - adjust as needed based on your models
                        estimated_layers = int((available_mem / model_size_gb) * 32)
                        n_gpu_layers = max(1, min(32, estimated_layers))
                        logger.info(f"Limited GPU layers to {n_gpu_layers} based on available VRAM ({available_mem:.2f}GB)")
            except Exception as e:
                logger.warning(f"Error calculating optimal GPU layers: {e}")
            
            # Load the model with optimized settings
            logger.info(f"Loading model with n_gpu_layers={n_gpu_layers}")
            self.model = Llama(
                model_path=self.model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=2048,
                n_batch=512,
                f16_kv=True,
                use_mlock=True,
                verbose=False  # Set to True for more detailed loading logs
            )
            
            self.progress_signal.emit(100)
            self.status_signal.emit("LLM model loaded")
            return True
        except Exception as e:
            logger.error(f"Error loading LLM model: {e}")
            return False
    
    def set_prompt(self, prompt):
        self.prompt = prompt
        
    def set_conversation_history(self, history):
        self.conversation_history = history
        
    def run(self):
        if not self.prompt or not self.model:
            return
            
        try:
            self.status_signal.emit("Generating response...")
            self.progress_signal.emit(10)
            
            # Format the prompt correctly for Qwen2.5 using its chat template format
            # Build a properly formatted prompt string with the required tokens
            prompt_text = "<|im_start|>system\n"
            prompt_text += self.system_prompt
            prompt_text += "<|im_end|>\n"
            
            # Add conversation history
            for i in range(0, len(self.conversation_history), 2):
                if i < len(self.conversation_history):
                    # User message
                    prompt_text += "<|im_start|>user\n"
                    prompt_text += self.conversation_history[i]
                    prompt_text += "<|im_end|>\n"
                
                if i+1 < len(self.conversation_history):
                    # Assistant message
                    prompt_text += "<|im_start|>assistant\n"
                    prompt_text += self.conversation_history[i+1]
                    prompt_text += "<|im_end|>\n"
            
            # Add current prompt
            prompt_text += "<|im_start|>user\n"
            prompt_text += self.prompt
            prompt_text += "<|im_end|>\n"
            
            # Add the assistant start token to prompt generation
            prompt_text += "<|im_start|>assistant\n"
            
            logger.info(f"Using formatted prompt template for Qwen2.5 with {len(self.conversation_history)//2} turn(s) of conversation")
            
            # Use complete instead of create_chat_completion for more direct control
            response = self.model.complete(
                prompt=prompt_text,
                max_tokens=self.max_tokens,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                repeat_penalty=1.1,
                stop=["<|im_end|>", "<|im_start|>"]  # Stop at these tokens
            )
            
            # Extract the generated text
            if response and 'choices' in response and len(response['choices']) > 0:
                text = response['choices'][0]['text'].strip()
                logger.info(f"Generated response: '{text}'")
                
                # Check if response is empty and provide fallback
                if not text:
                    text = "I understand. Is there anything specific you'd like to know more about?"
                    logger.warning(f"Empty response from LLM, using fallback: '{text}'")
            else:
                text = "I understand. How can I help you with that?"
                logger.warning(f"No valid response from LLM, using fallback: '{text}'")
            
            self.progress_signal.emit(100)
            self.status_signal.emit("Response generated")
            self.result_signal.emit(text)
            
        except Exception as e:
            logger.error(f"Error in text generation: {e}")
            # Provide a fallback response
            self.result_signal.emit("I apologize, but I encountered an error while generating a response. How can I help you?")


# Speech generation worker using CSM
class CSMWorker(QThread):
    result_signal = pyqtSignal(str, str)  # (text, audio_path)
    status_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    auth_needed_signal = pyqtSignal()  # Signal for authentication needed
    auth_completed_signal = pyqtSignal(bool)  # Signal to indicate auth is complete
    
    def __init__(self):
        super().__init__()
        self.text = ""
        self.assistant_voice_dir = ASSISTANT_VOICE_DIR
        self.user_voice_dir = USER_VOICE_DIR
        self.auth_event = threading.Event()  # Event for thread synchronization
        self.auth_success = False
        
    def set_text(self, text):
        self.text = text
        
    def set_voice_dirs(self, assistant_dir, user_dir=None):
        self.assistant_voice_dir = assistant_dir
        if user_dir:
            self.user_voice_dir = user_dir
        
    def load_model(self):
        """Check if CSM requirements are installed and prepare samples"""
        try:
            # Ensure voice sample directories exist
            os.makedirs(self.assistant_voice_dir, exist_ok=True)
            os.makedirs(self.user_voice_dir, exist_ok=True)
            
            # Prepare voice samples if needed
            assistant_samples = list(Path(self.assistant_voice_dir).glob('*.wav'))
            user_samples = list(Path(self.user_voice_dir).glob('*.wav'))
            
            if not assistant_samples:
                logger.warning(f"No assistant voice samples found in {self.assistant_voice_dir}")
                return False
                
            if not user_samples:
                logger.warning(f"No user voice samples found in {self.user_voice_dir}")
                self.create_dummy_user_samples()
                
            return True
        except Exception as e:
            logger.error(f"Error checking CSM setup: {e}")
            return False
    
    def create_dummy_user_samples(self):
        """Create dummy user voice samples if none exist"""
        try:
            sample_rate = 16000
            duration = 0.5  # seconds
            
            # Create simple dummy samples
            for i in range(5):
                # Create simple sine wave tone
                t = np.linspace(0, duration, int(sample_rate * duration), False)
                tone = np.sin(2 * np.pi * 440 * t) * 0.1  # 440 Hz tone at low volume
                audio = (tone * 32767).astype(np.int16)
                
                file_path = os.path.join(self.user_voice_dir, f"user_sample_{i+1}.wav")
                
                with wave.open(file_path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio.tobytes())
                
                logger.info(f"Created dummy user voice sample: {file_path}")
            
            logger.info(f"Created dummy user voice samples in {self.user_voice_dir}")
        except Exception as e:
            logger.error(f"Error creating dummy user samples: {e}")
            
    def on_auth_completed(self, success):
        """Handle authentication completion"""
        self.auth_success = success
        try:
            self.auth_event.set()  # Signal the waiting thread
        except Exception as e:
            logger.error(f"Error signaling auth completion: {e}")
    
    def run(self):
        """Generate speech using CSM"""
        try:
            # Import with proper handling for authentication
            from huggingface_hub import hf_hub_download, login
            
            try:
                import triton
                logger.info("Triton module found, will use for acceleration")
            except ImportError:
                logger.info("Triton module not found, will continue without it")
            
            self.status_signal.emit("Generating voice...")
            self.progress_signal.emit(20)
            
            # Try to download the model
            try:
                model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
                logger.info(f"Downloaded CSM model from HuggingFace")
            except Exception as e:
                if "401 Client Error" in str(e):
                    logger.info("Authentication required for HuggingFace Hub")
                    
                    # Signal main thread to handle authentication and wait for result
                    self.auth_event.clear()
                    self.auth_needed_signal.emit()
                    
                    # Wait for authentication to complete (with timeout)
                    auth_timeout = 60  # seconds
                    if not self.auth_event.wait(auth_timeout):
                        raise Exception("Authentication timed out")
                    
                    if not self.auth_success:
                        raise Exception("Authentication failed or was cancelled")
                    
                    # Try downloading again after authentication
                    model_path = hf_hub_download(repo_id="sesame/csm-1b", filename="ckpt.pt")
                else:
                    raise e
            
            from generator import load_csm_1b, Segment
            import torch
            import torchaudio
            
            # Load CSM model
            generator = load_csm_1b(model_path, "cuda")
            logger.info(f"Loaded CSM model, sample rate: {generator.sample_rate}")
            
            # Prepare context segments with interleaved samples from both speakers
            transcription_file = os.path.join(self.assistant_voice_dir, "transcriptions.json")
            if os.path.exists(transcription_file):
                with open(transcription_file, 'r') as f:
                    transcriptions = json.load(f)
                logger.info(f"Loaded transcriptions from {transcription_file}")
            else:
                transcriptions = {}
                logger.warning(f"No transcriptions file found at {transcription_file}")
            
            # Get all audio samples
            assistant_samples = list(Path(self.assistant_voice_dir).glob('*.wav'))
            user_samples = list(Path(self.user_voice_dir).glob('*.wav'))
            
            # Prepare segments in interleaved format as per CSM repo example
            segments = []
            audio_tensors = {}  # Cache audio tensors
            
            # First, load and process all audio files
            all_samples = []
            
            # For assistant samples, use actual transcriptions
            for file_path in assistant_samples:
                file_name = os.path.basename(file_path)
                text = transcriptions.get(file_name, f"Assistant sample {file_name}")
                all_samples.append((1, str(file_path), text))  # Speaker 1 = Assistant
            
            # For user samples, assign random utterances
            for i, file_path in enumerate(user_samples):
                # Use random utterance or fallback to general text
                text = random.choice(USER_UTTERANCES) if USER_UTTERANCES else f"User utterance {i+1}"
                all_samples.append((0, str(file_path), text))  # Speaker 0 = User
            
            # Shuffle to simulate natural conversation (but ensure we have a mix of speakers)
            random.shuffle(all_samples)
            
            # Make sure we start with alternating user and assistant samples for better context
            # Sort by speaker to ensure we have both types, then pick alternatingly
            user_clips = [s for s in all_samples if s[0] == 0]
            assistant_clips = [s for s in all_samples if s[0] == 1]
            
            # Create interleaved pattern (user, assistant, user, assistant...)
            interleaved_samples = []
            for i in range(max(len(user_clips), len(assistant_clips))):
                if i < len(user_clips):
                    interleaved_samples.append(user_clips[i])
                if i < len(assistant_clips):
                    interleaved_samples.append(assistant_clips[i])
            
            # Now process the interleaved samples
            for speaker, file_path, text in interleaved_samples:
                try:
                    # Load and process audio
                    if file_path not in audio_tensors:
                        audio, sample_rate = torchaudio.load(file_path)
                        
                        # Convert to mono if needed
                        if audio.shape[0] > 1:
                            audio = torch.mean(audio, dim=0, keepdim=True)
                        
                        # Remove channel dimension and resample if needed
                        audio = audio.squeeze(0)
                        if sample_rate != generator.sample_rate:
                            audio = torchaudio.functional.resample(
                                audio, 
                                orig_freq=sample_rate,
                                new_freq=generator.sample_rate
                            )
                            
                        audio_tensors[file_path] = audio
                    else:
                        audio = audio_tensors[file_path]
                    
                    # Create segment
                    segment = Segment(text=text, speaker=speaker, audio=audio)
                    segments.append(segment)
                    
                    logger.info(f"Added {'assistant' if speaker == 1 else 'user'} voice sample with text: '{text}'")
                except Exception as e:
                    logger.error(f"Error processing audio file {file_path}: {e}")
            
            logger.info(f"Created conversation context with {len(segments)} segments")
            
            # Log the first few samples to show the pattern
            for i, segment in enumerate(segments[:6]):
                speaker_type = "Assistant" if segment.speaker == 1 else "User"
                logger.info(f"Context segment {i+1}: {speaker_type} - '{segment.text}'")
                    
            # Generate speech
            self.progress_signal.emit(50)
            logger.info(f"Generating speech for: '{self.text}'")
            
            # Generate the audio
            audio = generator.generate(
                text=self.text,
                speaker=1,  # Assistant is always speaker 1
                context=segments,
                max_audio_length_ms=15000,
                temperature=0.8
            )
            
            # Save audio to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_file.close()
            
            torchaudio.save(temp_file.name, audio.unsqueeze(0).cpu(), generator.sample_rate)
            logger.info(f"Speech generated and saved to {temp_file.name}")
            
            self.progress_signal.emit(100)
            self.status_signal.emit("Voice generated")
            self.result_signal.emit(self.text, temp_file.name)
            
        except Exception as e:
            logger.error(f"Error in CSM speech generation: {e}")
            self.result_signal.emit(self.text, "")


# Response container to store assistant response with playback
class ResponseWidget(QWidget):
    """Widget to display assistant response with playback button"""
    
    def __init__(self, text, audio_path, parent=None):
        super().__init__(parent)
        self.text = text
        self.audio_path = audio_path
        self.audio_player = AudioPlayer(self)
        
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Response text
        self.text_label = QTextEdit()
        self.text_label.setReadOnly(True)
        self.text_label.setText(f"<b>Assistant:</b> {text}")
        layout.addWidget(self.text_label, 1)
        
        # Playback button
        self.play_button = QPushButton("▶")
        self.play_button.setMaximumWidth(30)
        self.play_button.clicked.connect(self.play_audio)
        layout.addWidget(self.play_button)
        
        self.setLayout(layout)
        
    def play_audio(self):
        """Play the audio response"""
        if self.audio_path and os.path.exists(self.audio_path):
            self.audio_player.play(self.audio_path)
            self.play_button.setEnabled(False)
            self.play_button.setText("⏹")
            
            # Re-enable button when playback finishes
            def on_playback_finished():
                self.play_button.setEnabled(True)
                self.play_button.setText("▶")
            
            self.audio_player.finished_signal.connect(on_playback_finished)
        else:
            logger.error(f"Audio file not found: {self.audio_path}")


class VoiceConversationApp(QMainWindow):
    """Main application window for Voice Conversation System"""
    
    # Define signals at class level
    setup_csm_signal = pyqtSignal()
    handle_llm_failure_signal = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice Conversation System")
        self.setGeometry(100, 100, 900, 700)
        
        # Set dark theme
        self.setup_dark_theme()
        
        # Initialize conversation history
        self.conversation_history = []
        
        # Create layout
        main_layout = QVBoxLayout()
        
        # Chat display
        self.chat_display = QWidget()
        self.chat_layout = QVBoxLayout()
        self.chat_layout.setAlignment(Qt.AlignTop)
        self.chat_layout.setSpacing(10)
        self.chat_display.setLayout(self.chat_layout)
        
        # Scroll area for chat
        from PyQt5.QtWidgets import QScrollArea
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.chat_display)
        scroll_area.setMinimumHeight(400)
        main_layout.addWidget(scroll_area)
        
        # Recording controls
        record_layout = QHBoxLayout()
        
        self.record_button = QPushButton("Start Recording")
        self.record_button.setMinimumHeight(60)
        self.record_button.setEnabled(False)  # Disabled until models are loaded
        self.record_button.clicked.connect(self.toggle_recording)
        record_layout.addWidget(self.record_button)
        
        self.send_button = QPushButton("Send")
        self.send_button.setMinimumHeight(60)
        self.send_button.setEnabled(False)  # Disabled until recording is done
        self.send_button.clicked.connect(self.send_recording)
        record_layout.addWidget(self.send_button)
        
        self.clear_button = QPushButton("Clear")
        self.clear_button.setMinimumHeight(60)
        self.clear_button.setEnabled(False)  # Disabled until recording is done
        self.clear_button.clicked.connect(self.clear_recording)
        record_layout.addWidget(self.clear_button)
        
        main_layout.addLayout(record_layout)
        
        # Other controls
        control_layout = QHBoxLayout()
        
        self.clear_chat_button = QPushButton("Clear Conversation")
        self.clear_chat_button.clicked.connect(self.clear_conversation)
        control_layout.addWidget(self.clear_chat_button)
        
        # Load models button
        self.load_models_button = QPushButton("Load Models")
        self.load_models_button.setMinimumHeight(50)
        self.load_models_button.clicked.connect(self.load_all_models)
        control_layout.addWidget(self.load_models_button)
        
        main_layout.addLayout(control_layout)
        
        # LLM Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("LLM Model:"))
        self.llm_path_edit = QLineEdit()
        self.llm_path_edit.setReadOnly(True)
        self.llm_path_edit.setPlaceholderText("No model selected")
        model_layout.addWidget(self.llm_path_edit)
        self.llm_select_button = QPushButton("Select")
        self.llm_select_button.clicked.connect(self.select_llm_model)
        model_layout.addWidget(self.llm_select_button)
        
        main_layout.addLayout(model_layout)
        
        # Voice samples directory selection
        voice_layout = QHBoxLayout()
        voice_layout.addWidget(QLabel("Voice Samples:"))
        self.voice_samples_path = QLineEdit()
        self.voice_samples_path.setReadOnly(True)
        self.voice_samples_path.setPlaceholderText(f"Default: ./{ASSISTANT_VOICE_DIR}")
        voice_layout.addWidget(self.voice_samples_path)
        self.voice_select_button = QPushButton("Select Folder")
        self.voice_select_button.clicked.connect(self.select_voice_samples_dir)
        voice_layout.addWidget(self.voice_select_button)
        
        main_layout.addLayout(voice_layout)
        
        # Transcribe samples button
        transcribe_layout = QHBoxLayout()
        self.transcribe_button = QPushButton("Transcribe Voice Samples")
        self.transcribe_button.clicked.connect(self.transcribe_voice_samples)
        transcribe_layout.addWidget(self.transcribe_button)
        
        main_layout.addLayout(transcribe_layout)
        
        # Audio visualization
        audio_layout = QVBoxLayout()
        audio_layout.addWidget(QLabel("Audio Level:"))
        self.waveform_widget = VoiceWaveformWidget()
        audio_layout.addWidget(self.waveform_widget)
        
        main_layout.addLayout(audio_layout)
        
        # Status and progress area
        status_layout = QVBoxLayout()
        
        # Status indicator
        self.status_label = QLabel("System Status: Please load models to begin")
        status_layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        status_layout.addWidget(self.progress_bar)
        
        main_layout.addLayout(status_layout)
        
        # Set main widget
        main_widget = QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Initialize workers
        self.audio_recorder = AudioRecorder()
        self.audio_recorder.finished_signal.connect(self.on_recording_finished)
        self.audio_recorder.audio_level_signal.connect(self.update_audio_level)
        self.audio_recorder.status_signal.connect(self.update_status)
        
        self.speech_recognition_worker = SpeechRecognitionWorker()
        self.speech_recognition_worker.result_signal.connect(self.on_transcription_complete)
        self.speech_recognition_worker.status_signal.connect(self.update_status)
        self.speech_recognition_worker.progress_signal.connect(self.update_progress)
        self.speech_recognition_worker.model_ready_signal.connect(self.on_whisper_model_ready)
        
        self.text_generation_worker = TextGenerationWorker()
        self.text_generation_worker.result_signal.connect(self.on_text_generation_complete)
        self.text_generation_worker.status_signal.connect(self.update_status)
        self.text_generation_worker.progress_signal.connect(self.update_progress)
        
        self.csm_worker = CSMWorker()
        self.csm_worker.result_signal.connect(self.on_speech_generation_complete)
        self.csm_worker.status_signal.connect(self.update_status)
        self.csm_worker.progress_signal.connect(self.update_progress)
        self.csm_worker.auth_needed_signal.connect(self.on_auth_needed)
        
        # Connect custom signals
        self.setup_csm_signal.connect(self.setup_csm)
        self.handle_llm_failure_signal.connect(self._handle_llm_load_failure)
        
        # Initialize flags
        self.is_recording = False
        self.models_loaded = False
        self.whisper_model_loaded = False
        self.current_recording_file = None
        self.load_progress_timer = QTimer()
        self.load_progress_current = 0
        
        # Add welcome message
        welcome_widget = QTextEdit()
        welcome_widget.setReadOnly(True)
        welcome_widget.setMaximumHeight(100)
        welcome_widget.append("<b>System:</b> Welcome to the Voice Conversation System!")
        welcome_widget.append("<b>System:</b> To begin, please select an LLM model and voice samples directory, then load all models.")
        self.chat_layout.addWidget(welcome_widget)
        
        # Ensure voice directories exist
        os.makedirs(ASSISTANT_VOICE_DIR, exist_ok=True)
        os.makedirs(USER_VOICE_DIR, exist_ok=True)
        
        # Perform system check
        success, message = check_system_resources()
        if not success:
            system_widget = QTextEdit()
            system_widget.setReadOnly(True)
            system_widget.setMaximumHeight(60)
            system_widget.append(f"<b>System Warning:</b> {message}")
            self.chat_layout.addWidget(system_widget)
        
        # Log application start
        logger.info("Voice Conversation System started")
    
    def setup_dark_theme(self):
        """Setup dark theme for the UI"""
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        
        QApplication.setPalette(dark_palette)
        QApplication.setStyle("Fusion")
    
    def select_llm_model(self):
        """Open file dialog to select GGUF LLM model"""
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("GGUF Models (*.gguf)")
        if file_dialog.exec_():
            model_path = file_dialog.selectedFiles()[0]
            self.llm_path_edit.setText(model_path)
            logger.info(f"Selected LLM model: {model_path}")
    
    def select_voice_samples_dir(self):
        """Open directory dialog to select voice samples directory"""
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.Directory)
        if dialog.exec_():
            dir_path = dialog.selectedFiles()[0]
            self.voice_samples_path.setText(dir_path)
            logger.info(f"Selected voice samples directory: {dir_path}")
    
    def update_audio_level(self, level):
        """Update the audio level visualization"""
        self.waveform_widget.update_level(level)
    
    def authenticate_huggingface(self):
        """Prompt for HuggingFace token and authenticate"""
        from huggingface_hub import login
        
        token, ok = QInputDialog.getText(self, "HuggingFace Login", 
                               "Please enter your HuggingFace token\n(Get it from https://huggingface.co/settings/tokens):")
        if ok and token:
            try:
                login(token=token)
                self.update_status("HuggingFace authenticated successfully")
                return True
            except Exception as e:
                logger.error(f"HuggingFace authentication failed: {e}")
                QMessageBox.warning(self, "Authentication Failed", 
                              f"Failed to authenticate with HuggingFace: {str(e)}")
        return False
    
    def on_auth_needed(self):
        """Handle authentication request from CSM worker"""
        success = self.authenticate_huggingface()
        self.csm_worker.on_auth_completed(success)
    
    def load_all_models(self):
        """Load all AI models in sequence"""
        # Check if LLM model is selected
        if not self.llm_path_edit.text():
            QMessageBox.warning(self, "Model Missing", "Please select an LLM model first.")
            return
        
        # Check if voice samples directory is set and has samples
        voice_dir = self.voice_samples_path.text() or ASSISTANT_VOICE_DIR
        voice_samples = list(Path(voice_dir).glob('*.wav'))
        if not voice_samples:
            reply = QMessageBox.question(self, "No Voice Samples", 
                                    f"No voice samples found in '{voice_dir}'. Voice synthesis requires samples.\n\nWould you like to select a different directory?",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                self.select_voice_samples_dir()
                return
            
        self.update_status("Loading models... This may take a while")
        self.progress_bar.setValue(0)
        self.load_models_button.setEnabled(False)
        
        # Free up CUDA memory
        try:
            torch.cuda.empty_cache()
        except:
            pass
        
        # Load models sequentially
        self.load_whisper_model()
    
    def load_whisper_model(self):
        """Load Whisper model"""
        self.update_status("Loading Whisper model...")
        
        # Load in thread
        self.speech_recognition_worker.load_model()
        
    def on_whisper_model_ready(self, success):
        """Called when Whisper model is ready"""
        if success:
            self.whisper_model_loaded = True
            # Continue with next model
            self.load_llm_model()
        else:
            self.update_status("Failed to load Whisper model")
            self.load_models_button.setEnabled(True)
    
    def load_llm_model(self):
        """Load LLM model with better error handling"""
        self.update_status("Loading LLM model...")
        self.progress_bar.setValue(30)
        
        # Set model path
        model_path = self.llm_path_edit.text()
        if not os.path.exists(model_path):
            self.update_status(f"Error: Model file not found at {model_path}")
            QMessageBox.critical(self, "Error", f"Model file not found at {model_path}")
            self.load_models_button.setEnabled(True)
            return
        
        # Check file size to warn about large models
        try:
            model_size_gb = os.path.getsize(model_path) / (1024**3)
            if model_size_gb > 10:
                reply = QMessageBox.question(
                    self, "Large Model", 
                    f"The selected model is {model_size_gb:.1f} GB. Loading may take a while and require "
                    f"significant memory. Continue?",
                    QMessageBox.Yes | QMessageBox.No, 
                    QMessageBox.Yes
                )
                if reply == QMessageBox.No:
                    self.update_status("Model loading cancelled by user")
                    self.load_models_button.setEnabled(True)
                    return
        except Exception as e:
            logger.warning(f"Couldn't check model size: {e}")
        
        self.text_generation_worker.set_model_path(model_path)
        
        # Set a customized system prompt
        self.text_generation_worker.set_system_prompt(
            "You are a helpful voice assistant with a friendly and conversational style. "
            "Speak naturally and concisely. Be helpful and informative while keeping "
            "responses reasonably brief for spoken conversation."
        )
        
        # Show a more detailed status message
        self.update_status(f"Loading LLM model {os.path.basename(model_path)}... This might take a minute")
        
        # Create a progress timer to update progress bar continuously during loading
        self.load_progress_timer = QTimer(self)  # Parent to self to ensure it stays on the main thread
        self.load_progress_timer.timeout.connect(self._update_load_progress)
        self.load_progress_current = 30
        self.load_progress_timer.start(100)  # Update every 100ms
        
        # Load in a separate thread
        loading_thread = threading.Thread(target=self._load_llm_model_thread)
        loading_thread.daemon = True
        loading_thread.start()

    def _update_load_progress(self):
        """Update progress bar during model loading"""
        if self.load_progress_current < 65:
            self.load_progress_current += 0.5
            self.progress_bar.setValue(int(self.load_progress_current))

    def _load_llm_model_thread(self):
        """Thread to load LLM model"""
        success = self.text_generation_worker.load_model()
        
        # Stop the progress timer using safe method
        self.load_progress_timer.stop()
        
        # Use signals to safely communicate between threads
        if success:
            # Continue to next step using signal
            self.setup_csm_signal.emit()
        else:
            # Handle failure using signal
            self.handle_llm_failure_signal.emit()

    def _handle_llm_load_failure(self):
        """Handle LLM model load failure"""
        self.update_status("Failed to load LLM model")
        QMessageBox.critical(self, "Error", "Failed to load LLM model. Please check the logs for details.")
        self.load_models_button.setEnabled(True)
    
    def setup_csm(self):
        """Set up CSM voice synthesis"""
        self.update_status("Setting up voice synthesis...")
        self.progress_bar.setValue(70)
        
        # Set voice directories
        voice_dir = self.voice_samples_path.text() or ASSISTANT_VOICE_DIR
        self.csm_worker.set_voice_dirs(voice_dir, USER_VOICE_DIR)
        
        # Load CSM
        if self.csm_worker.load_model():
            # All models loaded successfully
            self.models_loaded = True
            self.record_button.setEnabled(True)
            self.update_status("All models loaded. Ready!")
            self.progress_bar.setValue(100)
            self.load_models_button.setEnabled(True)
            
            # Show info about voice samples
            voice_dir = self.voice_samples_path.text() or ASSISTANT_VOICE_DIR
            voice_samples = list(Path(voice_dir).glob('*.wav'))
            
            # Add status message to chat
            status_widget = QTextEdit()
            status_widget.setReadOnly(True)
            status_widget.setMaximumHeight(100)
            status_widget.append(f"<b>System:</b> Found {len(voice_samples)} voice samples.")
            
            # Check if samples have transcriptions
            transcription_file = os.path.join(voice_dir, "transcriptions.json")
            if os.path.exists(transcription_file):
                status_widget.append("<b>System:</b> Transcriptions file found.")
            else:
                status_widget.append("<b>System:</b> <font color='orange'>No transcriptions file found. You should transcribe your voice samples.</font>")
            
            status_widget.append("<b>System:</b> System is ready for conversation. Press the 'Start Recording' button to start talking.")
            
            self.chat_layout.addWidget(status_widget)
        else:
            self.update_status("Failed to set up voice synthesis")
            self.load_models_button.setEnabled(True)
    
    def transcribe_voice_samples(self):
        """Transcribe voice samples in the selected directory"""
        if not self.whisper_model_loaded:
            if QMessageBox.question(self, "Load Whisper Model", 
                                  "Whisper model is not loaded. Load it now?",
                                  QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
                # Load Whisper model first
                def on_model_ready(success):
                    if success:
                        self.whisper_model_loaded = True
                        self.transcribe_voice_samples_with_whisper()
                    else:
                        QMessageBox.warning(self, "Error", "Failed to load Whisper model")
                
                # Disconnect previous connections if any
                try:
                    self.speech_recognition_worker.model_ready_signal.disconnect()
                except:
                    pass
                
                # Connect to the signal
                self.speech_recognition_worker.model_ready_signal.connect(on_model_ready)
                
                # Load the model
                self.update_status("Loading Whisper model...")
                self.speech_recognition_worker.load_model()
            return
        
        # Whisper model is loaded, proceed with transcription
        self.transcribe_voice_samples_with_whisper()
    
    def transcribe_voice_samples_with_whisper(self):
        """Transcribe voice samples using the loaded Whisper model"""
        voice_dir = self.voice_samples_path.text() or ASSISTANT_VOICE_DIR
        voice_samples = list(Path(voice_dir).glob('*.wav'))
        
        if not voice_samples:
            QMessageBox.warning(self, "No Samples", f"No voice samples found in {voice_dir}")
            return
        
        self.update_status(f"Transcribing {len(voice_samples)} voice samples...")
        
        # Set up batch transcription
        self.speech_recognition_worker.set_batch_files([str(path) for path in voice_samples])
        
        # Connect to batch transcription result
        def on_batch_complete(results_json):
            try:
                results = json.loads(results_json)
                
                # Save transcriptions to file
                transcription_file = os.path.join(voice_dir, "transcriptions.json")
                with open(transcription_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
                self.update_status("Transcription complete")
                QMessageBox.information(self, "Success", 
                                      f"Successfully transcribed {len(results)} voice samples.\nSaved to {transcription_file}")
                
                # Show a few examples in the chat widget
                examples_widget = QTextEdit()
                examples_widget.setReadOnly(True)
                examples_widget.setMaximumHeight(150)
                examples_widget.append("<b>System:</b> Voice sample transcriptions:")
                for i, (file, text) in enumerate(list(results.items())[:3]):
                    examples_widget.append(f"<b>Sample {i+1}:</b> {text}")
                if len(results) > 3:
                    examples_widget.append(f"<b>System:</b> ... and {len(results)-3} more samples")
                
                self.chat_layout.addWidget(examples_widget)
                
            except Exception as e:
                logger.error(f"Error processing transcription results: {e}")
                QMessageBox.warning(self, "Error", f"Error saving transcriptions: {str(e)}")
        
        # Disconnect previous connections if any
        try:
            self.speech_recognition_worker.result_signal.disconnect()
        except:
            pass
        
        # Connect to the result signal
        self.speech_recognition_worker.result_signal.connect(on_batch_complete)
        
        # Start transcription
        self.speech_recognition_worker.start()
    
    def toggle_recording(self):
        """Toggle recording start/stop"""
        if not self.models_loaded:
            QMessageBox.warning(self, "Models Not Loaded", "Please load all models before recording.")
            return
        
        if not self.is_recording:
            # Start recording
            self.is_recording = True
            self.record_button.setText("Stop Recording")
            self.send_button.setEnabled(False)
            self.clear_button.setEnabled(False)
            logger.info("Starting audio recording")
            
            # Create a temporary file for the recording
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_file.close()
            self.current_recording_file = temp_file.name
            
            # Configure recorder
            self.audio_recorder.set_auto_stop(False)  # Disable auto-stop
            self.audio_recorder.set_output_file(self.current_recording_file)
            
            # Start recording
            self.waveform_widget.clear_levels()
            self.audio_recorder.start()
        else:
            # Stop recording
            self.is_recording = False
            self.record_button.setText("Start Recording")
            logger.info("Stopping audio recording")
            self.audio_recorder.stop()
    
    def on_recording_finished(self, audio_file):
        """Handle completed recording"""
        if not audio_file:
            self.update_status("Recording failed or empty")
            return
            
        # Enable send/clear buttons
        self.send_button.setEnabled(True)
        self.clear_button.setEnabled(True)
        self.update_status("Recording complete. Press Send to process or Clear to discard.")
    
    def send_recording(self):
        """Process the current recording"""
        if not self.current_recording_file or not os.path.exists(self.current_recording_file):
            self.update_status("No recording available")
            return
            
        # Disable buttons while processing
        self.send_button.setEnabled(False)
        self.clear_button.setEnabled(False)
        
        # Process the recording
        self.process_audio(self.current_recording_file)
    
    def clear_recording(self):
        """Clear the current recording"""
        if self.current_recording_file and os.path.exists(self.current_recording_file):
            try:
                os.unlink(self.current_recording_file)
                logger.info(f"Deleted recording file: {self.current_recording_file}")
            except Exception as e:
                logger.error(f"Error deleting recording file: {e}")
                
        # Reset state
        self.current_recording_file = None
        self.send_button.setEnabled(False)
        self.clear_button.setEnabled(False)
        self.update_status("Recording cleared. Press Start Recording to begin again.")
    
    def process_audio(self, audio_file):
        """Process recorded audio file"""
        self.update_status("Transcribing speech...")
        self.progress_bar.setValue(0)
        logger.info(f"Processing recorded audio file: {audio_file}")
        
        # Reconnect the single transcription signal handler
        try:
            self.speech_recognition_worker.result_signal.disconnect()
        except:
            pass
        self.speech_recognition_worker.result_signal.connect(self.on_transcription_complete)
        
        # Start speech recognition
        self.speech_recognition_worker.set_audio_file(audio_file)
        self.speech_recognition_worker.start()
    
    def on_transcription_complete(self, text):
        """Handle completed transcription"""
        if not text.strip():
            self.update_status("No speech detected, try again")
            return
            
        self.update_status("Generating response...")
        self.progress_bar.setValue(0)
        
        # Display user message
        user_widget = QTextEdit()
        user_widget.setReadOnly(True)
        user_widget.setMaximumHeight(80)
        user_widget.append(f"<b>You:</b> {text}")
        self.chat_layout.addWidget(user_widget)
        
        logger.info(f"User said: '{text}'")
        
        # Update conversation history
        self.conversation_history.append(text)
        
        # Generate text response
        self.text_generation_worker.set_prompt(text)
        self.text_generation_worker.set_conversation_history(self.conversation_history)
        self.text_generation_worker.start()
    
    def on_text_generation_complete(self, text):
        """Handle completed text generation"""
        self.update_status("Generating speech...")
        self.progress_bar.setValue(0)
        
        logger.info(f"Assistant response: '{text}'")
        
        # Update conversation history
        self.conversation_history.append(text)
        
        # Generate speech
        self.csm_worker.set_text(text)
        self.csm_worker.start()
    
    def on_speech_generation_complete(self, text, audio_path):
        """Handle completed speech generation"""
        self.update_status("Response ready")
        
        if not audio_path:
            # Create response widget without audio
            response_widget = QTextEdit()
            response_widget.setReadOnly(True)
            response_widget.setMaximumHeight(100)
            response_widget.setText(f"<b>Assistant:</b> {text}")
            self.chat_layout.addWidget(response_widget)
            
            self.update_status("Ready (no audio available)")
            return
            
        logger.info(f"Speech generated successfully, saved to: {audio_path}")
        
        # Create response widget with playback
        response_widget = ResponseWidget(text, audio_path)
        self.chat_layout.addWidget(response_widget)
        
        # Play audio automatically
        response_widget.play_audio()
        
        self.update_status("Ready")
        self.progress_bar.setValue(0)
        
        # Re-enable recording
        self.record_button.setEnabled(True)
    
    def clear_conversation(self):
        """Clear the conversation history"""
        reply = QMessageBox.question(self, "Clear Conversation", 
                                    "Are you sure you want to clear the conversation history?",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            # Clear conversation history
            self.conversation_history = []
            
            # Clear chat widgets
            while self.chat_layout.count():
                item = self.chat_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            
            # Add welcome message back
            welcome_widget = QTextEdit()
            welcome_widget.setReadOnly(True)
            welcome_widget.setMaximumHeight(80)
            welcome_widget.append("<b>System:</b> Conversation history cleared.")
            welcome_widget.append("<b>System:</b> Press 'Start Recording' to begin a new conversation.")
            self.chat_layout.addWidget(welcome_widget)
            
            logger.info("Conversation history cleared")
    
    def update_progress(self, value):
        """Update the progress bar"""
        self.progress_bar.setValue(value)
    
    def update_status(self, status):
        """Update the status label"""
        self.status_label.setText(f"System Status: {status}")
        logger.info(f"Status: {status}")


if __name__ == "__main__":
    # Enable CUDA TF32 for faster inference
    try:
        # Fix OpenMP issue
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        
        # Configure CUDA
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Free up CUDA memory
        torch.cuda.empty_cache()
        
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"CUDA setup: {e}")
    
    # Create and launch the application
    app = QApplication(sys.argv)
    window = VoiceConversationApp()
    window.show()
    sys.exit(app.exec_())