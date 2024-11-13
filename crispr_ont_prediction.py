#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Dropout, Dense, Reshape, Lambda, Permute, Flatten, Input, Embedding, multiply, Conv1D, AveragePooling1D
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
import tensorflow.keras
import numpy as np
import pandas as pd


def make_data(seq):
  tok = Tokenizer(char_level=False, oov_token='N') 
  tok.fit_on_texts(['A', 'T', 'C', 'G'])
  return np.array(tok.texts_to_sequences(seq)).reshape(1, 24)



class AttentionLayer(Layer):
    def __init__(self, time_steps, **kwargs):
        self.time_steps = time_steps
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        self.dense_x = Dense(self.time_steps, kernel_initializer=RandomUniform(seed=2020))
        self.dense_g = Dense(self.time_steps, kernel_initializer=RandomUniform(seed=2020))
        self.attention_dense = Dense(self.time_steps, activation="softmax", use_bias=False)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        x, g = inputs
        x1 = K.permute_dimensions(x, (0, 2, 1))
        g1 = K.permute_dimensions(g, (0, 2, 1))

        x2 = Reshape((x1.shape[1], self.time_steps))(x1)
        g2 = Reshape((g1.shape[1], self.time_steps))(g1)

        x3 = self.dense_x(x2)
        g3 = self.dense_g(g2)

        x4 = tensorflow.keras.layers.add([x3, g3])
        a = self.attention_dense(x4)
        a_probs = Permute((2, 1))(a)
        output_attention_mul = multiply([x, a_probs])

        return output_attention_mul

    def compute_output_shape(self, input_shape):
        return input_shape[0]

def crispr_ont():
    dropout_rate = 0.4
    input = Input(shape=(24,))
    embedded = Embedding(7, 44, input_length=24)(input)

    conv1 = Conv1D(256, 5, activation="relu", name="conv1")(embedded)
    pool1 = AveragePooling1D(2)(conv1)
    drop1 = Dropout(dropout_rate)(pool1)

    conv2 = Conv1D(256, 5, activation="relu", name="conv2")(pool1)
    conv3 = Conv1D(256, 5, activation="relu", name="conv3")(drop1)

    attention_layer = AttentionLayer(6)
    x = attention_layer([conv3, conv2])

    my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    weight_1 = Lambda(lambda x: x * 0.2)
    weight_2 = Lambda(lambda x: x * 0.8)

    flat1 = Flatten()(pool1)
    flat2 = Flatten()(x)
    flat = my_concat([weight_1(flat1), weight_2(flat2)])

    dense1 = Dense(128, kernel_regularizer=regularizers.l2(1e-4), bias_regularizer=regularizers.l2(1e-4),
                   activation="relu", name="dense1")(flat)
    drop3 = Dropout(dropout_rate)(dense1)

    dense2 = Dense(64, kernel_regularizer=regularizers.l2(1e-4), bias_regularizer=regularizers.l2(1e-4),
                   activation="relu", name="dense2")(drop3)
    drop4 = Dropout(dropout_rate)(dense2)

    dense3 = Dense(32, activation="relu", name="dense3")(drop4)
    drop5 = Dropout(dropout_rate)(dense3)

    output = Dense(1, activation="linear", name="output")(drop5)

    model = Model(inputs=[input], outputs=[output])
    return model


if __name__ == '__main__':
    model = crispr_ont()

    print("Loading weights for the models")
    model.load_weights("crispr_ont.h5")
    
    data_path = "test_ont.csv"
    data = pd.read_csv(data_path)
    x_test = make_data('N' + data["sgRNA"][0])
    print("data: ", x_test)

    y_pred = model.predict([x_test])
    print("prediction of ONT: ", y_pred)
