import pickle
import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import random
import tkinter as tk
from tkinter import ttk
import threading
import sys  # For application exit

# Initialize lemmatizer for preprocessing
lemmatizer = WordNetLemmatizer()

# Configuration for file paths and thresholds
CONFIG = {
    "MODEL_PATH": 'models/chatbot_model.keras',
    "WORDS_PATH": 'models/words.pkl',
    "CLASSES_PATH": 'models/classes.pkl',
    "INTENTS_PATH": 'data/intents.json',
    "ERROR_THRESHOLD": 0.25
}

# Load the necessary files (model, words, classes, intents)
def load_files(config):
    """
    Loads the trained model, vocabulary, classes, and intents data.
    Includes error handling for missing files or incorrect paths.

    Args:
        config (dict): Configuration dictionary with file paths.

    Returns:
        tuple: Loaded model, words list, classes list, and intents data.
    """
    try:
        model = load_model(config['MODEL_PATH'])
        words = pickle.load(open(config['WORDS_PATH'], 'rb'))
        classes = pickle.load(open(config['CLASSES_PATH'], 'rb'))
        with open(config['INTENTS_PATH'], 'r') as file:
            intents = json.load(file)
        return model, words, classes, intents
    except FileNotFoundError as e:
        sys.exit(f"Critical Error: Missing file - {e}")
    except Exception as e:
        sys.exit(f"Critical Error while loading resources: {e}")

# Load resources
model, words, classes, intents = load_files(CONFIG)

# Track chatbot context for conversation flow
context = {"active": None}

# Preprocessing user input
def clean_up_sentence(sentence):
    """
    Tokenizes and lemmatizes the user's input sentence.

    Args:
        sentence (str): User's input sentence.

    Returns:
        list: Lemmatized words from the input sentence.
    """
    sentence = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence if word.isalnum()]

def bag_of_words(sentence, words):
    """
    Converts the user's input into a bag-of-words representation.

    Args:
        sentence (str): User's input sentence.
        words (list): Vocabulary list.

    Returns:
        np.array: Bag-of-words representation of the input sentence.
    """
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if word in sentence_words else 0 for word in words]
    return np.array(bag)

def predict_class(sentence, model, words, classes, error_threshold):
    """
    Predicts the intent of the user's input based on the trained model.
    Supports multi-intent detection by returning intents above the confidence threshold.

    Args:
        sentence (str): User's input sentence.
        model: Trained chatbot model.
        words: Vocabulary list.
        classes: List of intent classes.
        error_threshold (float): Confidence threshold for predictions.

    Returns:
        list: Predicted intents and their probabilities.
    """
    bow = bag_of_words(sentence, words)
    probabilities = model.predict(np.array([bow]))[0]
    results = [(i, prob) for i, prob in enumerate(probabilities) if prob > error_threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{'intent': classes[result[0]], 'probability': f"{result[1]:.2f}"} for result in results]

def get_response(intents_list, intents):
    """
    Generates a response based on the predicted intent.
    Leverages context tracking for follow-up responses.

    Args:
        intents_list (list): Predicted intents and probabilities.
        intents (dict): Intents data from the JSON file.

    Returns:
        str: Chatbot's response.
    """
    global context
    if intents_list:
        tag = intents_list[0]['intent']
        for intent in intents['intents']:
            if intent['tag'] == tag:
                # Check if the intent requires context tracking
                if "context_set" in intent:
                    context['active'] = intent['context_set']
                if context['active'] and "context_filter" in intent and intent['context_filter'] != context['active']:
                    return "Could you clarify that further?"
                return random.choice(intent['responses'])
    return "I'm sorry, I didn't understand that. Could you rephrase?"

# GUI functions
def send_message():
    """
    Handles user input and triggers the bot's response generation.
    """
    user_message = user_input.get().strip()
    if user_message:
        add_message("You", user_message, "#a1d6e2")
        user_input.set("")
        if user_message.lower() in ["quit", "end chat", "bye", "goodbye"]:
            goodbye_and_exit()
        elif user_message.lower() == "help":
            add_message("Bot", "Here are some commands you can try:\n- Hi\n- Help\n- Track Order\n- What's the weather?\n- Locations\n- Goodbye", "#ffe3e3")
        else:
            threading.Thread(target=generate_response, args=(user_message,)).start()

def generate_response(user_message):
    """
    Simulates the bot's typing and generates a response.
    Handles multiple intents in the user's input.

    Args:
        user_message (str): User's input message.
    """
    add_typing_indicator()
    intents_list = predict_class(user_message, model, words, classes, CONFIG['ERROR_THRESHOLD'])
    if len(intents_list) > 1:
        bot_response = "I detected multiple intents. Here's what I can address:\n"
        bot_response += "\n".join([f"- {intent['intent']} (confidence: {intent['probability']})" for intent in intents_list])
    else:
        bot_response = get_response(intents_list, intents)
    remove_typing_indicator()
    add_message("Bot", bot_response, "#ffe3e3")

def goodbye_and_exit():
    """
    Displays a goodbye message and exits the application.
    """
    add_message("Bot", "Goodbye! Have a great day!", "#ffe3e3")
    root.after(2000, root.destroy)

def add_message(sender, message, color):
    """
    Displays a message in the chat log.

    Args:
        sender (str): The sender of the message (e.g., "You", "Bot").
        message (str): The message text.
        color (str): The background color for the avatar.
    """
    chat_log.insert(tk.END, f"\n{sender}:\n", "sender")
    add_avatar(chat_log, color)
    chat_log.insert(tk.END, f" {message}\n", "message")
    chat_log.see(tk.END)

def add_avatar(widget, color):
    """
    Adds a colored circle as an avatar for the sender.

    Args:
        widget: The chat log widget.
        color (str): The color of the avatar circle.
    """
    canvas = tk.Canvas(widget, width=20, height=20, bg="white", bd=0, highlightthickness=0)
    canvas.create_oval(5, 5, 20, 20, fill=color, outline="")
    widget.window_create(tk.END, window=canvas)

def add_typing_indicator():
    """
    Displays a typing indicator in the chat log.
    """
    chat_log.insert(tk.END, "\nBot is typing...", "typing")
    chat_log.see(tk.END)

def remove_typing_indicator():
    """
    Removes the typing indicator from the chat log.
    """
    start_index = chat_log.index(f"1.0 linestart")
    while True:
        index = chat_log.search("Bot is typing...", start_index, tk.END, nocase=True)
        if not index:
            break
        line_end = chat_log.index(f"{index} lineend")
        chat_log.delete(index, line_end)
        start_index = index
    chat_log.see(tk.END)

def clear_chat():
    """
    Clears the chat log.
    """
    chat_log.delete(1.0, tk.END)

# GUI setup
root = tk.Tk()
root.title("Creative Chatbot")
root.geometry("600x700")
root.configure(bg="#f2f2f2")

# Chat log setup
chat_frame = tk.Frame(root, bg="#ffffff", bd=1)
chat_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

chat_log = tk.Text(chat_frame, wrap=tk.WORD, state=tk.NORMAL, bg="#fdfdfd", font=("Arial", 12), bd=0)
chat_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar = ttk.Scrollbar(chat_frame, command=chat_log.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
chat_log['yscrollcommand'] = scrollbar.set

chat_log.tag_config("sender", foreground="#333333", font=("Arial", 10, "bold"))
chat_log.tag_config("message", foreground="#555555", font=("Arial", 12))
chat_log.tag_config("typing", foreground="#999999", font=("Arial", 12, "italic"))

# Input frame setup
input_frame = tk.Frame(root, bg="#f0f0f0")
input_frame.pack(padx=10, pady=10, fill=tk.X)

user_input = tk.StringVar()
input_box = tk.Entry(input_frame, textvariable=user_input, font=("Arial", 14), fg='black', bg="lightyellow", bd=1)
input_box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

send_button = tk.Button(input_frame, text="Send", command=send_message, bg="#4CAF50", fg="#6600ba", font=("Arial", 12))
send_button.pack(side=tk.RIGHT)

clear_button = tk.Button(input_frame, text="Clear", command=clear_chat, bg="#f44336", fg="#6600ba", font=("Arial", 12))
clear_button.pack(side=tk.RIGHT, padx=(0, 5))

# Start the GUI event loop
root.mainloop()
