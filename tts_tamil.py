from transformers import AutoModel
import numpy as np
import soundfile as sf
import os
import torch
import logging
from huggingface_hub import login

# 🔹 Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 🔹 Hugging Face Authentication
HF_TOKEN = "<Your huggingface token>"
login(HF_TOKEN)

# 🔹 Load IndicF5 model
repo_id = "ai4bharat/IndicF5"

try:
    model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
    logging.info("✅ Model loaded successfully!")
except Exception as e:
    logging.error(f"❌ Error loading model: {e}")
    exit(1)

# 🔹 Tamil + English Mixed Text for Synthesis (Tanglish)
text = "வணக்கம்! Today is a beautiful day. Let's enjoy and make good memories."

# 🔹 Tamil reference audio (Ensuring correct Tamil pronunciation)
ref_audio_path = "IndicF5/prompts/TAM_F_HAPPY_00001.wav"
ref_text = "வணக்கம்! இன்று ஒரு மகிழ்ச்சியான நாள்."

# 🔹 Check if reference audio exists
if not os.path.exists(ref_audio_path):
    logging.error(f"❌ Reference audio file NOT found: {ref_audio_path}")
    exit(1)
logging.info(f"📂 Reference audio found: {ref_audio_path}")

# 🔹 Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"🚀 Using device: {device}")
model.to(device)

# 🔹 Generate speech with explicit Tamil setting
try:
    logging.info("🎤 Generating speech using IndicF5 (Tamil + English Mode)...")
    
    # 🔹 Run model inference with Tamil language setting
    audio_data = model(text, ref_audio_path=ref_audio_path, ref_text=ref_text, language="ta")

    # 🔹 Validate if model returned proper audio
    if audio_data is None or len(audio_data) == 0:
        logging.error("❌ No audio data generated!")
        exit(1)

    # 🔹 Function to normalize audio correctly
    def normalize_audio(audio_array):
        audio_array = np.asarray(audio_array)
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32) / np.max(np.abs(audio_array))
        return audio_array

    # 🔹 Normalize and save output
    normalized_audio = normalize_audio(audio_data)

    output_dir = "IndicF5/output"
    output_filename = "tamil_english_speech.wav"
    output_path = os.path.join(output_dir, output_filename)

    # 🔹 Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 🔹 Save the output with correct sampling rate
    sample_rate = 24000
    sf.write(output_path, normalized_audio, samplerate=sample_rate)

    # 🔹 Check if file exists
    if os.path.exists(output_path):
        logging.info(f"✅ Tamil + English mixed speech generated successfully!")
        logging.info(f"📂 File saved at: {output_path}")
        logging.info(f"💾 File size: {os.path.getsize(output_path) / 1024:.2f} KB")
    else:
        logging.error("❌ Tamil + English speech file was NOT created!")

except Exception as e:
    logging.error(f"❌ Error generating speech: {e}")
