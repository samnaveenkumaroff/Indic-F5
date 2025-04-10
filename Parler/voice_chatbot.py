# -*- coding: utf-8 -*-

import gradio as gr
import whisper  # Corrected import for Whisper
from groq import Groq
from tempfile import NamedTemporaryFile
import torch
from tts_tamil import TamilTTS  # Replace ParlerTTS with TamilTTS

# Initialize Groq API Client
client = Groq(
    api_key="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",
)

# Load the Whisper model
model = whisper.load_model("base")

def speech_to_text(audio_path):
    transcription = model.transcribe(audio_path)["text"]
    return transcription

def generate_response(text):
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": text}
        ],
        model="qwen-2.5-32b",
    )
    return chat_completion.choices[0].message.content

def text_to_speech(text):
    tts = TamilTTS(text)  # Using TamilTTS from tts_tamil.py
    output_audio = NamedTemporaryFile(suffix=".wav", delete=False)
    tts.save(output_audio.name)  # Save generated audio
    return output_audio.name

def chatbot_pipeline(audio_path):
    try:
        # Step 1: Convert speech to text
        text_input = speech_to_text(audio_path)

        # Step 2: Get response from LLaMA model
        response_text = generate_response(text_input)

        # Step 3: Convert response text to Tamil speech
        response_audio_path = text_to_speech(response_text)

        return response_text, response_audio_path

    except Exception as e:
        return str(e), None

# Create Gradio Interface
iface = gr.Interface(
    fn=chatbot_pipeline,
    inputs=gr.Audio(type="filepath", label="Speak"),
    outputs=[
        gr.Textbox(label="Response Text"),
        gr.Audio(label="Response Audio")
    ],
    title="Real-Time Tamil Voice-to-Voice Chatbot"
)

iface.launch()
