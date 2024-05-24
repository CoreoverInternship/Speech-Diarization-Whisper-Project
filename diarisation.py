import whisperx
import gc
from pyannote.audio import Model, Inference
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import io
from pydub import AudioSegment
from pydub.utils import make_chunks
import speech_recognition as sr
import pandas as pd
import os
# ========================================================================================================
def record_audio(duration, sample_rate=44100, output_file="output.wav"):
   
    print("Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2, dtype=np.int16)
    sd.wait() 
    print("Recording complete. Saving file...")

    write(output_file, sample_rate, audio_data)  
    print(f"File saved as {output_file}")


def mp3_to_wav(mp3_file_path):
    # Extract the directory and base name
    directory, base_name = os.path.split(mp3_file_path)
    # Create a new file name by replacing the .mp3 extension with .wav
    wav_base_name = os.path.splitext(base_name)[0] + '.wav'
    # Combine the directory and new base name to form the new file path
    wav_file_path = os.path.join(directory, wav_base_name)
    
    # Load the MP3 file
    audio = AudioSegment.from_mp3(mp3_file_path)
    
    # Export as WAV
    audio.export(wav_file_path, format="wav")
    
    

def speech_to_text(audio_chunk):
    # recgoniser = sr.Recognizer()

    audio_chunk_file = io.BytesIO()
    audio_chunk.export(audio_chunk_file, format="wav")
    audio_chunk_file.seek(0)


    # # print("got here")
    # with sr.AudioFile(audio_chunk_file) as source:
    #     audio = recgoniser.record(source)
    # try:
    #     text = recgoniser.recognize_google(audio)
    #     return text
    # except Exception as e:
    #     print("Exception: " + str(e))
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    from datasets import load_dataset

    from transformers import pipeline
    
   
    pipe = pipeline("automatic-speech-recognition", model="quinnb/whisper-Large-v3-hindi")


    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "quinnb/whisper-Large-v3-hindi"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=15,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    # dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
    # sample = dataset[0]["audio"]
    audio = audio_chunk_file.read()
    audio_chunk_file.seek(0)

    result = pipe(audio, return_timestamps=True)
    return result["text"]
    # print(result["text"])

# ========================================================================================================
def diarize(audiopath):
    with open('diarise.txt', 'w', encoding='utf-8') as f:
        pass
    audio = whisperx.load_audio(audiopath)



    diarization = whisperx.DiarizationPipeline(use_auth_token="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",device=device)

    diarized_segments = diarization(audio)

    # print(diarized_segments)

    usable_data_segments = pd.DataFrame(diarized_segments)

    # print(usable_data_segments)

    for index, row in usable_data_segments.iterrows():
        start_time = row['start'] *1000
        # print(start_time)
        end_time = row['end'] *1000

        audioForConversion = AudioSegment.from_wav(audiopath)

        audio_chunk = audioForConversion[start_time:end_time]

        text = speech_to_text(audio_chunk)

        if text == None:
            text = "-"
        elif text == "Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.":
            text = ""
        with open('diarise.txt', 'a', encoding='utf-8') as f:
            f.write(row['speaker'] + ": " + text + "\n")

device = 'cpu'#for running on GPU
batch_size = 2
compute_type = 'float32'
reordOrFile = input("use file input or record? f for file r  for record: ")
if(reordOrFile == "f"):
    file = input("what file?\n")
    diarize("audioFiles/"+file)
elif(reordOrFile == "r"):
    duration = input("for how long?")
    record_audio(duration)
    diarize("output.wav")
    print("diarizeing")

    

audiopath = 'car.wav'

# processAudio(audiopath)

# model = Model.from_pretrained("pyannote/segmentation-3.0",use_auth_token="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")


# audio = whisperx.load_audio(audiopath)

# model = whisperx.load_model("large-v2",device,compute_type=compute_type)
# v2 model is better is not specifying language



    # print(row['speaker'] + ": " + text) 
    # print(end_time)

# print(diarized_segments)
# print(diarized_segments[0]["start"])
# print(diarized_segments[0]["end"])
# for segment in diarized_segments:
#     print(segment)
#     print(segment[2])
   

# for segment in diarized_segments:
#     chunk_size = segment["end"] - segment["start"]
#     chunk = make_chunks(audio, chunk_size)

#     speech_to_text(chunk, segment["speaker"])




# print(diarized_segments)

# speakerAndText = whisperx.assign_word_speakers(diarized_segments,transcript)

# print(speakerAndText["segments"])

# transcript code ----------------------------
# model = whisperx.load_model("large-v2",device,compute_type=compute_type)
# # v2 model is better is not specifying language

# audio = whisperx.load_audio(audiopath)

# transcript = model.transcribe(audio, batch_size=batch_size)

# for segment in transcript["segments"]:
#     print(segment["text"])
#     print('\n')
# transcript code ----------------------------




# print(transcript["segments"])

