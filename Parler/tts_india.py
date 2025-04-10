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

# Supported Languages
SUPPORTED_LANGUAGES = {...}  # Keeping the dictionary same as your original
LANGUAGE_PROMPTS = {...}  # Keeping the dictionary same as your original

def generate_response(text, language):
    """Generate response using AI while enforcing the correct language output."""
    if language not in SUPPORTED_LANGUAGES:
        return f"❌ Unsupported language: {language}."
    
    lang_prompt = LANGUAGE_PROMPTS.get(language, "Respond in the selected language.")
    final_prompt = f"{lang_prompt}\n\nUser: {text}\nAI:"
    
    try:
        # Ensure `client` is defined before using this
        completion = client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-70b-instruct",
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0.5,
            top_p=1,
            max_tokens=1024,
        )
        response_text = completion.choices[0].message.content.strip()
        
        # Validate correct language output (improved check)
        if language in SUPPORTED_LANGUAGES and not response_text.startswith(LANGUAGE_PROMPTS[language][:5]):
            response_text = f"{LANGUAGE_PROMPTS[language]}\n{response_text}"

        return response_text

    except Exception as e:
        return f"❌ Error in generating response: {str(e)}"

def generate_speech(text, language):
    """Convert AI-generated text into speech in the selected Indian language."""
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}. Choose from {list(SUPPORTED_LANGUAGES.keys())}")
    
    lang_code = SUPPORTED_LANGUAGES[language]
    description = f"{language} voice with a neutral tone and normal speed."
    
    # Tokenize inputs
    description_input_ids = tokenizer(description, return_tensors="pt").to(device)
    prompt_input_ids = tokenizer(text, return_tensors="pt").to(device)
    
    # Generate Speech
    with torch.no_grad():
        generation = model.generate(
            input_ids=description_input_ids.input_ids,
            attention_mask=description_input_ids.attention_mask,
            prompt_input_ids=prompt_input_ids.input_ids,
            prompt_attention_mask=prompt_input_ids.attention_mask
        )
    
    # Extract raw speech tensor (ensure correct shape)
    audio_arr = generation.squeeze(0).cpu().numpy().astype(np.float32)  
    
    if len(audio_arr.shape) > 1:
        audio_arr = audio_arr.flatten()
    
    # Save output as WAV file
    output_audio = NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(output_audio.name, audio_arr, model.config.sampling_rate, subtype="PCM_16")
    
    return output_audio.name
