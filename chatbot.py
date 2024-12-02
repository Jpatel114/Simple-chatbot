import nltk
import json
import pickle
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

model = load_model('models/chatbot_model.keras')
words = pickle.load(open('models/words.pkl', 'rb'))
classes = pickle.load(open('models/classes.pkl', 'rb'))
with open('data/intents.json', 'r') as file:
    intents = json.load(file)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    return [{'intent': classes[i], 'probability': r} for i, r in enumerate(res) if r > ERROR_THRESHOLD]

def get_response(intents_list):
    tag = intents_list[0]['intent']
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return np.random.choice(intent['responses'])
    return "I didn't understand that."

if __name__ == "__main__":
    print("Chatbot is running. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        intents_list = predict_class(user_input)
        response = get_response(intents_list)
        print(f"Bot: {response}")
