# -*- coding: utf-8 -*-
import gradio as gr
import whisper
from groq import Groq
import torch
from tts_tamilcpu import tamil_tts  # Correct function name

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize Groq API Client
client = Groq(
    api_key="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",
)

# Load the Whisper model on GPU (if available)
print(f"⏳ Loading Whisper model on {device}...")
model = whisper.load_model("base").to(device)
print("✅ Whisper model loaded successfully!")

def speech_to_text(audio_path):
    """Convert speech to text using Whisper."""
    try:
        transcription = model.transcribe(audio_path)["text"]
        return transcription
    except Exception as e:
        return f"❌ Error in speech-to-text: {str(e)}"

def generate_response(text):
    """Generate AI response using Qwen-2.5-32B."""
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": text}],
            model="qwen-2.5-32b",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"❌ Error in generating response: {str(e)}"

def chatbot_pipeline(audio_path):
    """Complete pipeline: Speech-to-text → AI response → Text-to-Speech."""
    try:
        # Step 1: Convert speech to text
        print("🔄 Converting speech to text...")
        text_input = speech_to_text(audio_path)
        if "Error" in text_input:
            return text_input, None  # Return error message if STT fails
        print(f"📝 Transcription: {text_input}")

        # Step 2: Get AI-generated response
        print("🤖 Generating AI response...")
        response_text = generate_response(text_input)
        if "Error" in response_text:
            return response_text, None  # Return error message if LLM fails
        print(f"💬 AI Response: {response_text}")

        # Step 3: Convert response to Tamil speech
        print("🔊 Generating Tamil speech...")
        response_audio_path = tamil_tts(response_text)

        return response_text, response_audio_path

    except Exception as e:
        return f"❌ Unexpected Error: {str(e)}", None

# Create Gradio Interface
iface = gr.Interface(
    fn=chatbot_pipeline,
    inputs=gr.Audio(type="filepath", label="🎤 Speak"),
    outputs=[
        gr.Textbox(label="💬 Response Text"),
        gr.Audio(label="🔊 Response Audio"),
    ],
    title="🔴 Real-Time Tamil Voice-to-Voice Chatbot",
    description="🎙️ Speak in Tamil, get AI-generated responses as text & voice!"
)

if __name__ == "__main__":
    iface.launch()
