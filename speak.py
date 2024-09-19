import os
from playsound import playsound
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()


def speak(content):

    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="nova",
        input=content,
    ) as response:
        response.stream_to_file("speech.mp3")
    # Play the audio file
    playsound("speech.mp3")
    # Delete the audio file
    os.remove("speech.mp3")


speak("Heyy... i love you..")
