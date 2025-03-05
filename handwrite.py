# -*- coding: utf-8 -*-
"""handwrite

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Wj_TvRlm5jW_tkF5jvNvNTcdLcjLaYU2
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import random
import os

# Load dataset (Extended handwritten-like text)
text = """The quick brown fox jumps over the lazy dog.
Handwriting generation is fun! AI can create amazing things.
This is an example of a character-level RNN learning patterns."""

# Create character mapping
chars = sorted(list(set(text)))
char_to_index = {c: i for i, c in enumerate(chars)}
index_to_char = {i: c for i, c in enumerate(chars)}

# Prepare sequences
sequence_length = 10
dataX = []
dataY = []
for i in range(len(text) - sequence_length):
    seq_in = text[i:i + sequence_length]
    seq_out = text[i + sequence_length]
    dataX.append([char_to_index[char] for char in seq_in])
    dataY.append(char_to_index[seq_out])

# Reshape and normalize
dataX = np.array(dataX)
dataY = to_categorical(dataY, num_classes=len(chars))

# Build the RNN model
model = Sequential([
    Embedding(len(chars), 8, input_length=sequence_length),
    LSTM(128, return_sequences=True),
    LSTM(128),
    Dense(len(chars), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(dataX, dataY, epochs=100, batch_size=8)

# Function for temperature-based text generation
def sample_with_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Generate text with temperature
def generate_text(seed_text, length=200, temperature=0.8):
    result = seed_text
    for _ in range(length):
        x_input = np.array([[char_to_index[char] for char in result[-sequence_length:]]])
        x_input = pad_sequences(x_input, maxlen=sequence_length)
        prediction = model.predict(x_input, verbose=0)[0]
        next_index = sample_with_temperature(prediction, temperature)
        next_char = index_to_char[next_index]
        result += next_char
    return result

# Generate and print text
print("Generated Text:")
print(generate_text("Handwriting", length=200, temperature=0.8))