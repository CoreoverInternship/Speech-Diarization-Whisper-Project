import whisperx
import gc
from pyannote.audio import Model, Inference

import io
from pydub import AudioSegment
from pydub.utils import make_chunks
import speech_recognition as sr
import pandas as pd

# ========================================================================================================

def speech_to_text(chunk):
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


    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

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
    audio = audio_chunk_file

    result = pipe(audio, return_timestamps=True)
    print(result["chunks"])

# ========================================================================================================

device = 'cuda'#for running on GPU
batch_size = 4
compute_type = 'float16'

audiopath = 'charm.wav'

# processAudio(audiopath)

# model = Model.from_pretrained("pyannote/segmentation-3.0",use_auth_token="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")


# audio = whisperx.load_audio(audiopath)

# model = whisperx.load_model("large-v2",device,compute_type=compute_type)
# v2 model is better is not specifying language

audio = whisperx.load_audio(audiopath)

# transcript = model.transcribe(audio, batch_size=batch_size)

# transcribedText = transcript["segments"]
# # for segment in transcript["segments"]:
# #     print(segment["text"]) transcribe result is transcript[segments]
# #     print('\n')

# print(transcribedText)

diarization = whisperx.DiarizationPipeline(use_auth_token="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",device=device)

diarized_segments = diarization(audio)

print(diarized_segments)

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
    with open('diarise.txt', 'a') as f:
        f.write(row['speaker'] + ": " + text + "\n")

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

