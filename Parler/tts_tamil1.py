import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import numpy as np
from tempfile import NamedTemporaryFile

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Model
model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(device)
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

def generate_tamil_speech(text):
    """Convert Tamil text to speech and save as a WAV file."""
    description = "Jaya speaks with a neutral tone and normal speed."

    # Tokenize inputs
    description_input_ids = description_tokenizer(description, return_tensors="pt").to(device)
    prompt_input_ids = tokenizer(text, return_tensors="pt").to(device)

    # Generate Speech
    with torch.no_grad():
        generation = model.generate(
            input_ids=description_input_ids.input_ids.to(device),
            attention_mask=description_input_ids.attention_mask.to(device),
            prompt_input_ids=prompt_input_ids.input_ids.to(device),
            prompt_attention_mask=prompt_input_ids.attention_mask.to(device)
        )

    # Convert tensor to numpy
    audio_arr = generation[0].cpu().numpy().astype(np.float32)

    # Save output as WAV file
    output_audio = NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(output_audio.name, audio_arr, model.config.sampling_rate)

    return output_audio.name
