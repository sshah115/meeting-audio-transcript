from tqdm import tqdm
import whisper
import pyaudio
import wave
import keyboard
import os
import sys
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

def get_script_directory():
    if getattr(sys, 'frozen', False):
        # If the code is running as a bundle (e.g., as an executable)
        return os.path.dirname(sys.executable)
    else:
        # If the code is running as a script
        return os.path.dirname(os.path.abspath(__file__))

def record_and_transcribe():
    # Initialize the Whisper model
    model = whisper.load_model("base", download_root = "C:\\Users\\shail.shah\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\whisper")

    # Get the script directory
    script_dir = get_script_directory()
    # print(get_script_directory())
   
    # Set the filename for recording
    recording_filename = os.path.join(script_dir, "Recording.wav")

    # Set the recording parameters
    format = pyaudio.paInt16
    channels = 1
    rate = 16000
    chunk = 1024

    # Initialize PyAudio and open the microphone stream
    p = pyaudio.PyAudio()
    stream = p.open(format=format,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)

    print("Recording... Press Ctrl+Q to stop recording.")

    # Start recording audio and save it to a file
    frames = []
    try:
        while True:
                data = stream.read(chunk)
                frames.append(data)
                if keyboard.is_pressed('ctrl+q'):  # Check if Ctrl+Q is pressed
                    break
    except KeyboardInterrupt:
        pass

    print("Recording completed.")

    # Stop and close the microphone stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded audio to a WAV file
    wf = wave.open(recording_filename, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b"".join(frames))
    wf.close()

    # Transcribe the recorded audio
    result = model.transcribe(recording_filename)
    transcribed_text = result["text"]

    # Print the transcribed text
    print("Transcribed Text: " + transcribed_text)

    # Save the transcribed text to a text file
    output_filename = os.path.join(script_dir, "Transcribed_Text.txt")
    with open(output_filename, "w") as f:
        f.write(transcribed_text)

    # Load the T5 model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
       
    max_length = 100
    
    # Check if GPU is available; if not, use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Preprocess the text and encode it as input for the model
    input_text = "summarize: " + transcribed_text
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    
    # Generate a summary
    summary = model.generate(input_ids, max_length=max_length)
    
    # Decode the summary
    summary_text = tokenizer.decode(summary[0], skip_special_tokens=True)

    # Print the summary
    print("Summary: " + summary_text)

    # Save the summary to a text file
    summary_filename = os.path.join(script_dir, "Summary.txt")
    with open(summary_filename, "w") as f:
        f.write(summary_text)        

if __name__ == "__main__":
    record_and_transcribe()
