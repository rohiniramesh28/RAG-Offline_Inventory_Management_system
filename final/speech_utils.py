import speech_recognition as sr
import pyttsx3
import time  # For adding delay

# Initialize once, reuse engine and recognizer for efficiency
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Set speech rate (optional)

recognizer = sr.Recognizer()

def listen_to_user(timeout=5, phrase_time_limit=7):
    with sr.Microphone() as source:
        print("Listening... Please speak now.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        except sr.WaitTimeoutError:
            print("Listening timed out while waiting for phrase to start.")
            return None

    try:
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        time.sleep(1.2)  # Small delay before responding
        print("Sorry, I could not understand. Please try again.")
        speak_text("Sorry, I could not understand. Please try again.")
        return None
    except sr.RequestError:
        print("Speech service error. Check your internet connection.")
        speak_text("Speech service error. Check your internet connection.")
        return None

def speak_text(text):
    if not text:
        print("No text to speak.")
        return
    engine.say(text)
    engine.runAndWait()
