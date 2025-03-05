import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.models import Sequential
import os

# 1. Load and preprocess the data
def load_data(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read().lower()
        return text
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

def preprocess_text(text):
    chars = sorted(list(set(text)))
    char_to_index = {c: i for i, c in enumerate(chars)}
    index_to_char = {i: c for c, i in char_to_index.items()}
    return chars, char_to_index, index_to_char

def create_sequences(text, seq_length, char_to_index):
    input_chars = [char_to_index[char] for char in text]
    sequences = []
    next_chars = []
    for i in range(0, len(input_chars) - seq_length):
        seq = input_chars[i:i + seq_length]
        next_char = input_chars[i + seq_length]
        sequences.append(seq)
        next_chars.append(next_char)
    return np.array(sequences), np.array(next_chars)

# 2. Build the RNN model
def build_model(vocab_size, embedding_dim, rnn_units, batch_size, seq_length):
    model = Sequential([
        Embedding(vocab_size, embedding_dim,  input_length=seq_length),
        LSTM(rnn_units, return_sequences=True, batch_input_shape=(batch_size, seq_length, embedding_dim), stateful=True, recurrent_initializer='glorot_uniform'),
        LSTM(rnn_units, stateful=True, recurrent_initializer='glorot_uniform'),
        Dense(vocab_size)
    ])
    return model

# 3. Train the model
def train_model(model, sequences, next_chars, epochs, batch_size, char_to_index):

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    model.fit(sequences, next_chars, epochs=epochs, batch_size=batch_size)

# 4. Generate text
def generate_text(model, start_string, char_to_index, index_to_char, num_chars_to_generate, seq_length):
    input_eval = [char_to_index[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []

    model.reset_states()
    for i in range(num_chars_to_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        text_generated.append(index_to_char[predicted_id])
        input_eval = tf.expand_dims([predicted_id], 0)

    return start_string + ''.join(text_generated)

# Main execution
filepath = 'handwrite_text.txt'
text = load_data(filepath)

if text:
    seq_length = 100
    batch_size = 64
    embedding_dim = 256
    rnn_units = 1024
    epochs = 20

    chars, char_to_index, index_to_char = preprocess_text(text)
    vocab_size = len(chars)

    sequences, next_chars = create_sequences(text, seq_length, char_to_index)

    model = build_model(vocab_size, embedding_dim, rnn_units, batch_size, seq_length)
    train_model(model, sequences, next_chars, epochs, batch_size, char_to_index)

    start_string = "Hello, "
    generated_text = generate_text(model, start_string, char_to_index, index_to_char, 500, seq_length)
    print(generated_text)