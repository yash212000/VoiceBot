import openai
import sounddevice as sd
import numpy as np
import time
from scipy.io.wavfile import write
from playsound import playsound

# Prompt user to enter OpenAI API key
OPENAI_API_KEY = input("Enter your OpenAI API key: ").strip()
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Audio settings
SAMPLE_RATE = 44100
DURATION = 5  # Adjust recording duration per interaction

def record_audio(filename="input.wav", duration=DURATION, sample_rate=SAMPLE_RATE):
    """Records audio from the microphone and saves it as a WAV file."""
    print("Listening...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
    sd.wait()
    write(filename, sample_rate, audio_data)
    return filename

def transcribe_audio(filename):
    """Converts speech to text using OpenAI's Whisper model."""
    with open(filename, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=audio_file,
            language="en"  # Force English transcription
        )
    return transcription.text.strip()

def generate_response(prompt):
    """Generates chatbot response using OpenAI GPT-4 with personal knowledge."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": """
                You are a chatbot designed to respond exactly as I would.
                Here are some key details about me:
                - Life Story: I am an ML engineer, I like traveling a lot, born in India, love playing badminton.
                - Superpower: Solving problems using AI
                - Areas of Growth: MLOps, Multimodal models
                - Misconceptions: That I am very serious
                - How you push your boundaries: By working smart and hard, Following a disciplined routine.

                When answering, consider these traits wherever applicable and respond as I would and keep it in 2-3 short sentences unless details are asked.
            """},
            {"role": "user", "content": prompt}
        ]

    )
    return response.choices[0].message.content.strip()

def text_to_speech(text, filename="response.mp3"):
    """Converts text to speech using OpenAI's TTS."""
    with client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="nova",
            input=text,
    ) as response:
        response.stream_to_file(filename)
    return filename

def play_audio(filename):
    playsound(filename)

def main():
    """Main loop for continuous real-time conversation."""
    print("AI Voice Chatbot Started. Speak to begin...")
    while True:
        input_audio = record_audio()
        user_text = transcribe_audio(input_audio)
        print(f"You: {user_text}")

        if user_text.lower() in ["exit", "quit", "stop"]:
            print("Exiting chat...")
            break

        ai_response = generate_response(user_text)
        print(f"AI: {ai_response}")

        response_audio = text_to_speech(ai_response)
        play_audio(response_audio)

        time.sleep(1)  # Short pause before next round

if __name__ == "__main__":
    main()