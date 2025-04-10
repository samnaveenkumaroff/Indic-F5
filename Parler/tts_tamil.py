import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

# Use CPU or GPU if available
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load Model
model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(device)
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

# Tamil TTS
prompt = "காருண்யா பல்கலைக்கழகம் இவ்வுலகிலேயே சிறந்த பல்கலைக்கழகம் காலத்தினால் செய்த நன்றி சிறிதெனினும் ஞானத்தின் மாணப் பெரிது எந்நன்றி கொன்றார்க்கும் உய்வுண்டாம் உய்வில்லை செய்நன்றி கொன்ற மகற்கு ணம் நாடி குற்றமும் நாடி எவற்றையும் மேகை நாடி மிக்க கொளல் உடுக்கை இழந்தவன் கைபோல ஆங்கே இடுக்கண் களைவதாம் நட்பு தோன்றின் புகழோடு தோன்றுக அகிலா தோன்றலின் தோன்றாமை நன்று அடக்கம் அமரருள் உய்க்கும் வணங்காமை பாலில் வைத்து விடும்"

description = "Jaya speaks with a neutral tone and normal speed. The recording is clear, high-quality, and close to the microphone."

description_input_ids = description_tokenizer(description, return_tensors="pt").to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").to(device)

# Generate Speech
generation = model.generate(
    input_ids=description_input_ids.input_ids,
    attention_mask=description_input_ids.attention_mask,
    prompt_input_ids=prompt_input_ids.input_ids,
    prompt_attention_mask=prompt_input_ids.attention_mask
)

# Save output
audio_arr = generation.cpu().numpy().squeeze()
sf.write("tamil_tts_output.wav", audio_arr, model.config.sampling_rate)

print("Tamil speech generated: tamil_tts_output.wav")
