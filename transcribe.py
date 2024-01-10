#%%
# setup
import whisperx
import sys
# import gc

device = "cuda"
audio_file = sys.argv[1]
output_file = sys.argv[2]
hf_token = sys.argv[3]
batch_size = 16
compute_type = "float16"
# %%
# 0. Load file
audio = whisperx.load_audio(audio_file)

# %%
# 1. Transcribe with original whisper (batched)
# TODO allow for model specification
model = whisperx.load_model("large-v2", device, compute_type=compute_type)
result = model.transcribe(audio, batch_size=batch_size)
print(result["segments"]) # before alignment

# %%
# 2. Align Whisper output
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

# print(result["segments"]) # after alignment
# %%
# 3. Assign speaker labels
diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)

# add min/max number of speakers if known
# TODO allow define num speakers
diarize_segments = diarize_model(audio)
diarize_model(audio, min_speakers=2, max_speakers=2)

result = whisperx.assign_word_speakers(diarize_segments, result)
print(diarize_segments)
print(result["segments"]) # segments are now assigned speaker IDs

# %%
# 5. Writeout
with open(output_file, 'w') as f:
    for s in result["segments"]:
        f.write(f'{s["speaker"]}: {s["text"]}\n')