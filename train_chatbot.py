import nltk
import numpy as np
import json
import pickle
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

nltk.download('punkt')
nltk.download('wordnet')

nltk.download('all')
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

lemmatizer = WordNetLemmatizer()

# Load intents
with open('data/intents.json', 'r') as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

# Tokenize and lemmatize data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = sorted(set(lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words))
classes = sorted(set(classes))

pickle.dump(words, open('models/words.pkl', 'wb'))
pickle.dump(classes, open('models/classes.pkl', 'wb'))

# Prepare training data
training = []
output_empty = [0] * len(classes)
for doc in documents:
    bag = [1 if w in doc[0] else 0 for w in words]
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

train_x = np.array([i[0] for i in training])
train_y = np.array([i[1] for i in training])

# Build the model
model = Sequential([
    Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(train_y[0]), activation='softmax')
])

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save('models/chatbot_model.keras')
