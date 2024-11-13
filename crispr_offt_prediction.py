#!/usr/bin/env python3

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow.keras
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.layers import multiply, Reshape, Permute, BatchNormalization, Dense, Dropout, Lambda, Flatten, Conv1D, Embedding, Input
from tensorflow.keras.models import *
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd


def loadData(filepath):
    """
    Load data from a file where each line contains a sequence followed by numerical features.
    Returns a numpy array where each row is a sequence of numerical features.
    """
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            sequence = parts[0]  # The sequence part
            features = np.array(parts[2:], dtype=np.float32)  # The numerical features part
            
            # Process the sequence into a numerical representation if needed
            # Here assuming that the model expects the sequence to be preprocessed
            # For now, just append features assuming that sequence preprocessing is handled separately
            data.append(features)
    
    return np.array(data)

VOCAB_SIZE = 16
EMBED_SIZE = 90
MAXLEN = 23
TIME_STEPS = 11

def attention(x, g, TIME_STEPS, dense1, dense2, dense3):
    """
    inputs.shape = (batch_size, time_steps, input_dim)
    """
    input_dim = int(x.shape[2])
    x1 = K.permute_dimensions(x, (0, 2, 1))
    g1 = K.permute_dimensions(g, (0, 2, 1))

    x2 = Reshape((input_dim, TIME_STEPS))(x1)
    g2 = Reshape((input_dim, TIME_STEPS))(g1)

    x3 = dense1(x2)
    g3 = dense2(g2)
    x4 = tensorflow.keras.layers.add([x3, g3])
    a = dense3(x4)
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

    # Define Dense layers outside of the attention function
    dense_time_steps = Dense(TIME_STEPS, kernel_initializer=RandomUniform(seed=2020))
    dense_g = Dense(TIME_STEPS, kernel_initializer=RandomUniform(seed=2020))
    dense_attention = Dense(TIME_STEPS, activation="softmax", use_bias=False)

    x = Lambda(lambda x: attention(x[0], x[1], 11, dense_time_steps, dense_g, dense_attention), output_shape=(11, 80))([conv11, batchnor3])

    flat = Flatten()(x)
    dense1 = Dense(40, activation="relu", name="dense1")(flat)
    drop1 = Dropout(0.2)(dense1)

    dense2 = Dense(20, activation="relu", name="dense2")(drop1)
    drop2 = Dropout(0.2)(dense2)

    output = Dense(2, activation="softmax", name="dense3")(drop2)
    model = Model(inputs=[input], outputs=[output])
    return model



if __name__ == '__main__':
    model = crispr_offt()

    print("Loading weights for the models")
    model.load_weights("crispr_offt.h5")
    encoded_file = "encoded_test_offt.txt"
        
    xtest = loadData(encoded_file)
    print("xtest: ", type(xtest), ", tartalma: ", xtest)
    print("Predicting on test data")
    y_pred = model.predict(xtest[:, :23])
    print("prediction of OFFT: ", y_pred)