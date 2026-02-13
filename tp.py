import speech_recognition as sr
import pyttsx3
import datetime
import json

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 0.9)  # Volume level

# Initialize speech recognition
recognizer = sr.Recognizer()

# Baymax's knowledge base
health_tips = {
    'headache': 'Rest in a quiet room. Stay hydrated and consider pain relief medication.',
    'fever': 'Your temperature is elevated. Rest, drink fluids, and monitor your condition.',
    'fatigue': 'Get adequate sleep, eat balanced meals, and stay hydrated.',
    'cough': 'Stay hydrated, use throat lozenges, and avoid irritants.',
    'stomach pain': 'Eat light foods, stay hydrated, and rest. Seek medical attention if severe.'
}

def speak(text):
    """Convert text to speech"""
    print(f"Baymax: {text}")
    engine.say(text)
    engine.runAndWait()

def listen():
    """Listen to user voice input"""
    try:
        with sr.Microphone() as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            print(f"You: {text}")
            return text.lower()
    except sr.UnknownValueError:
        speak("I did not understand. Please repeat.")
        return ""
    except sr.RequestError:
        speak("I cannot access the microphone or internet connection.")
        return ""

def get_greeting():
    """Generate time-based greeting"""
    hour = datetime.datetime.now().hour
    if hour < 12:
        return "Good morning. I am Baymax, your personal healthcare companion."
    elif hour < 18:
        return "Good afternoon. How can I assist you today?"
    else:
        return "Good evening. I am here to help with your health."

def check_health(symptom):
    """Check health based on symptom"""
    for key, advice in health_tips.items():
        if key in symptom:
            return f"I have detected {key}. {advice}"
    return "I do not have information about that symptom. Please consult a healthcare professional."

def get_time():
    """Get current time"""
    now = datetime.datetime.now()
    return f"The current time is {now.strftime('%I:%M %p')}"

def get_date():
    """Get current date"""
    now = datetime.datetime.now()
    return f"Today is {now.strftime('%A, %B %d, %Y')}"

def process_command(command):
    """Process user commands"""
    
    if 'hello' in command or 'hi' in command:
        return "Hello. I am Baymax. I am here to help you stay healthy."
    
    elif 'time' in command:
        return get_time()
    
    elif 'date' in command:
        return get_date()
    
    elif 'health' in command or 'symptom' in command or 'pain' in command or 'sick' in command or 'hurt' in command:
        speak("What symptoms are you experiencing?")
        symptom = listen()
        return check_health(symptom)
    
    elif 'water' in command:
        return "You should drink water regularly. Aim for 8 glasses per day."
    
    elif 'sleep' in command or 'rest' in command:
        return "Getting 7-9 hours of sleep is essential for your health."
    
    elif 'exercise' in command or 'workout' in command:
        return "Regular physical activity is important. Aim for at least 30 minutes daily."
    
    elif 'heart rate' in command:
        return "A normal resting heart rate is between 60 and 100 beats per minute."
    
    elif 'temperature' in command:
        return "Normal body temperature is approximately 98.6 degrees Fahrenheit."
    
    elif 'goodbye' in command or 'bye' in command or 'exit' in command:
        return "Goodbye. Please take care of yourself. I will be here when you need me."
    
    else:
        return "I am not sure how to help with that. Would you like health advice?"

def main():
    """Main function to run Baymax voice assistant"""
    speak(get_greeting())
    
    while True:
        command = listen()
        
        if not command:
            continue
        
        response = process_command(command)
        speak(response)
        
        if 'goodbye' in command or 'bye' in command or 'exit' in command:
            break

if __name__ == "__main__":
    print("=" * 50)
    print("BAYMAX VOICE ASSISTANT v1.0")
    print("=" * 50)
    print("\nStarting voice assistant...\n")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nBaymax: Goodbye. Stay healthy!")
        speak("Goodbye. Stay healthy!")