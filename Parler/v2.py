import gradio as gr
import whisper
import torch
from openai import OpenAI
from tempfile import NamedTemporaryFile
from tts_tamil1 import generate_tamil_speech  # Use Tamil TTS from `tts_tamil1.py`

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Whisper model on GPU if available
print("⏳ Loading Whisper model...")
model = whisper.load_model("base").to(device)
print("✅ Whisper model loaded successfully!")

# Initialize NVIDIA API client for Nemotron
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

def speech_to_text(audio_path):
    """Convert speech to text using Whisper."""
    try:
        transcription = model.transcribe(audio_path)["text"]
        return transcription
    except Exception as e:
        return f"❌ Error in speech-to-text: {str(e)}"

def generate_response(text):
    """Generate response using NVIDIA Nemotron model."""
    try:
        completion = client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-70b-instruct",
            messages=[{"role": "user", "content": text}],
            temperature=0.5,
            top_p=1,
            max_tokens=1024,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"❌ Error in generating response: {str(e)}"

def chatbot_pipeline(audio_path):
    """Complete pipeline: Speech-to-text → AI response → Tamil Speech."""
    try:
        print("🔄 Converting speech to text...")
        text_input = speech_to_text(audio_path)
        print(f"📝 Transcription: {text_input}")

        print("🤖 Generating AI response...")
        response_text = generate_response(text_input)
        print(f"💬 AI Response: {response_text}")

        print("🔊 Generating Tamil speech...")
        response_audio_path = generate_tamil_speech(response_text)  # Tamil TTS

        return text_input, response_text, response_audio_path

    except Exception as e:
        return "", str(e), None

# Create Gradio Interface
with gr.Blocks(title="🔴 Tamil Voice Chatbot with Nemotron") as iface:
    gr.Markdown("# 🎙️ Tamil Voice Assistant (Nemotron AI)")

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="🎤 Speak")
            submit_btn = gr.Button("🚀 Submit")

        with gr.Column():
            text_input = gr.Textbox(label="📝 Transcription")
            text_output = gr.Textbox(label="💬 AI Response")
            audio_output = gr.Audio(label="🔊 Tamil Speech")

    submit_btn.click(
        fn=chatbot_pipeline,
        inputs=[audio_input],
        outputs=[text_input, text_output, audio_output]
    )

# Launch Gradio App
if __name__ == "__main__":
    iface.launch()
