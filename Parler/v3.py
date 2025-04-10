import gradio as gr
import whisper
import torch
from openai import OpenAI
from tts_india import generate_speech  # Updated for multi-language TTS

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load Whisper model on selected device
print("⏳ Loading Whisper model...")
model = whisper.load_model("base").to(device)
print("✅ Whisper model loaded successfully!")

# Initialize NVIDIA API client for Nemotron
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

# Supported Languages
SUPPORTED_LANGUAGES = {
    "Assamese": "as", "Bengali": "bn", "Gujarati": "gu", "Hindi": "hi",
    "Kannada": "kn", "Kashmiri": "ks", "Konkani": "gom", "Maithili": "mai",
    "Malayalam": "ml", "Manipuri": "mni", "Marathi": "mr", "Nepali": "ne",
    "Odia": "or", "Punjabi": "pa", "Sanskrit": "sa", "Sindhi": "sd",
    "Tamil": "ta", "Telugu": "te", "Urdu": "ur", "Bodo": "brx",
    "Dogri": "doi", "Santali": "sat", "Meitei": "mni-Mtei"
}

def speech_to_text(audio_path):
    """Convert speech to text using Whisper."""
    try:
        transcription = model.transcribe(audio_path)["text"]
        return transcription
    except Exception as e:
        return f"❌ Error in speech-to-text: {str(e)}"

def generate_response(text, language):
    """Generate response using NVIDIA Nemotron model while enforcing strict language constraints."""
    if language not in SUPPORTED_LANGUAGES:
        return f"❌ Unsupported language: {language}."
    
    lang_prompt = f"Respond strictly in {language}. Do not use any other language."
    final_prompt = f"{lang_prompt}\n\nUser: {text}\nAI:" 
    
    try:
        completion = client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-70b-instruct",
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0.5,
            top_p=1,
            max_tokens=1024,
        )
        response_text = completion.choices[0].message.content.strip()
        return response_text
    except Exception as e:
        return f"❌ Error in generating response: {str(e)}"

def chatbot_pipeline(audio_path, language):
    """Complete pipeline: Speech-to-text → AI response → Multi-language Speech."""
    try:
        print("🔄 Converting speech to text...")
        text_input = speech_to_text(audio_path)
        print(f"📝 Transcription: {text_input}")

        print("🤖 Generating AI response...")
        response_text = generate_response(text_input, language)
        print(f"💬 AI Response: {response_text}")

        print("🔊 Generating speech...")
        response_audio_path = generate_speech(response_text, SUPPORTED_LANGUAGES[language])  # Multi-language TTS

        return text_input, response_text, response_audio_path

    except Exception as e:
        return "", str(e), None

# Create Gradio Interface
with gr.Blocks(title="🔴 Multilingual Voice Chatbot with Nemotron") as iface:
    gr.Markdown("# 🎙️ AI Voice Assistant (Nemotron + Whisper)")

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="🎤 Speak")
            language_input = gr.Dropdown(
                choices=list(SUPPORTED_LANGUAGES), label="🌎 Choose Language", value="Hindi"
            )
            submit_btn = gr.Button("🚀 Submit")

        with gr.Column():
            text_input = gr.Textbox(label="📝 Transcription")
            text_output = gr.Textbox(label="💬 AI Response")
            audio_output = gr.Audio(label="🔊 Generated Speech")

    submit_btn.click(
        fn=chatbot_pipeline,
        inputs=[audio_input, language_input],
        outputs=[text_input, text_output, audio_output]
    )

# Launch Gradio App
if __name__ == "__main__":
    iface.launch()
