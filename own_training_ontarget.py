#!/usr/bin/env python3

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Embedding, Conv1D, AveragePooling1D, Dropout, Flatten, 
                                     Dense, Lambda, Reshape, Permute, multiply, BatchNormalization)
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras import regularizers


# Load and preprocess the dataset
def load_data(filename):
    df = pd.read_csv(filename)
    
    # Tokenize the sgRNA sequences
    tokenizer = Tokenizer(char_level=True)  # Character-level tokenization
    tokenizer.fit_on_texts(df['sgRNA'])
    X = np.array(tokenizer.texts_to_sequences(df['sgRNA']))
        
    y = df['Normalized efficacy'].values
    return X, y, tokenizer

# Split data into train, validation, and test sets
def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

def attention(x, g, TIME_STEPS, attention_dense1, attention_dense2, attention_dense3):
    """
    x and g are 3D tensors of shape (batch_size, time_steps, input_dim)
    """
    input_dim = int(x.shape[2])

    x1 = K.permute_dimensions(x, (0, 2, 1))
    g1 = K.permute_dimensions(g, (0, 2, 1))

    x2 = Reshape((input_dim, TIME_STEPS))(x1)
    g2 = Reshape((input_dim, TIME_STEPS))(g1)

    x3 = attention_dense1(x2)
    g3 = attention_dense2(g2)

    x4 = tf.keras.layers.add([x3, g3])

    a = attention_dense3(x4)  # Attention weights
    a_probs = Permute((2, 1))(a)
    output_attention_mul = multiply([x, a_probs])
    return output_attention_mul


def crispr_offt():
    input = Input(shape=(23,))
    embedded = Embedding(VOCAB_SIZE, EMBED_SIZE, input_length=MAXLEN)(input)

    conv1 = Conv1D(20, 5, activation="relu", name="conv1")(embedded)
    batchnor1 = BatchNormalization()(conv1)

    conv2 = Conv1D(40, 5, activation="relu", name="conv2")(batchnor1)
    batchnor2 = BatchNormalization()(conv2)

    conv3 = Conv1D(80, 5, activation="relu", name="conv3")(batchnor2)
    batchnor3 = BatchNormalization()(conv3)

    conv11 = Conv1D(80, 9, name="conv11")(batchnor1)

    # Define attention-related Dense layers
    attention_dense1 = Dense(11, kernel_initializer=RandomUniform(seed=2020))
    attention_dense2 = Dense(11, kernel_initializer=RandomUniform(seed=2020))
    attention_dense3 = Dense(11, activation="softmax", use_bias=False)

    # Use Lambda layer with attention function
    x = Lambda(lambda inputs: attention(inputs[0], inputs[1], 11, attention_dense1, attention_dense2, attention_dense3))([conv11, batchnor3])

    flat = Flatten()(x)
    dense1 = Dense(40, activation="relu", name="dense1")(flat)
    drop1 = Dropout(0.2)(dense1)

    dense2 = Dense(20, activation="relu", name="dense2")(drop1)
    drop2 = Dropout(0.2)(dense2)

    output = Dense(2, activation="softmax", name="dense3")(drop2)
    model = Model(inputs=[input], outputs=[output])
    return model




# Load data
X, y, tokenizer = load_data('on_tarrget_efficacy.csv')
print("X: ----------", X)
print("y: ----------", y)


# Split data
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

# Compile and train the model
model = crispr_ont()
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f'Test MAE: {test_mae}')
