from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

model_id = "quinnb/whisper-Large-v3-hindi"
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, force_download=True)
processor = AutoProcessor.from_pretrained(model_id, force_download=True)