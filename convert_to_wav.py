import os
import sys
import glob
from pathlib import Path

# Function to check if a command is available
def is_command_available(command):
    import shutil
    return shutil.which(command) is not None

# Try to import required modules
try:
    from pydub import AudioSegment
    pydub_available = True
except ImportError:
    pydub_available = False
    print("Warning: pydub not installed. Installing required packages...")
    import subprocess
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "pydub"], check=True)
        print("pydub installed successfully.")
        try:
            from pydub import AudioSegment
            pydub_available = True
        except ImportError:
            print("Failed to import pydub after installation.")
    except Exception as e:
        print(f"Failed to install pydub: {e}")

# Check if ffmpeg is available
ffmpeg_available = is_command_available('ffmpeg')
if not ffmpeg_available and pydub_available:
    print("Warning: ffmpeg not found in PATH but is required by pydub.")
    print("Please install ffmpeg: https://ffmpeg.org/download.html")

def convert_mp3_to_wav(mp3_file, output_dir=None):
    """Convert an MP3 file to WAV format with proper settings for CSM"""
    if not pydub_available:
        print(f"Cannot convert {mp3_file}: pydub not available")
        return None
        
    try:
        print(f"Converting {mp3_file}...")
        sound = AudioSegment.from_mp3(mp3_file)
        
        # Convert to mono and 24kHz sample rate (CSM's native rate)
        sound = sound.set_channels(1)
        sound = sound.set_frame_rate(24000)
        
        # Determine output filename
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.basename(mp3_file)
            base_name = os.path.splitext(filename)[0]
            wav_path = os.path.join(output_dir, f"{base_name}.wav")
        else:
            wav_path = os.path.splitext(mp3_file)[0] + ".wav"
            
        # Export to WAV
        sound.export(wav_path, format="wav")
        print(f"Converted to {wav_path}")
        return wav_path
    except Exception as e:
        print(f"Error converting {mp3_file}: {e}")
        return None

def main():
    print("HAL 9000 Voice Sample Converter")
    print("-------------------------------")
    
    if not pydub_available:
        print("Error: pydub module is required but not available.")
        print("Please install pydub using: pip install pydub")
        return
        
    if not ffmpeg_available:
        print("Warning: ffmpeg not found. Conversions may fail.")
        print("Please install ffmpeg: https://ffmpeg.org/download.html")
    
    # Determine voice samples directory
    default_dir = "hal_voice_samples"
    voice_dir = input(f"Enter voice samples directory [default: {default_dir}]: ").strip()
    if not voice_dir:
        voice_dir = default_dir
        
    if not os.path.exists(voice_dir):
        print(f"Directory {voice_dir} not found. Creating it...")
        try:
            os.makedirs(voice_dir)
        except Exception as e:
            print(f"Error creating directory: {e}")
            return
    
    # Find MP3 files
    mp3_files = list(Path(voice_dir).glob("*.mp3"))
    if not mp3_files:
        print(f"No MP3 files found in {voice_dir}")
        return
        
    print(f"Found {len(mp3_files)} MP3 files to convert")
    
    # Convert all MP3 files
    converted = 0
    for mp3_file in mp3_files:
        wav_path = convert_mp3_to_wav(str(mp3_file))
        if wav_path:
            converted += 1
            
    print(f"Conversion complete: {converted}/{len(mp3_files)} files converted")
    
if __name__ == "__main__":
    main()