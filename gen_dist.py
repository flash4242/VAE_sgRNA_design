#!/usr/bin/env python3


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import wandb
from wandb.integration.keras import WandbCallback

from crispr_ont_prediction import crispr_ont, make_data


data = pd.read_csv('sgrna_hela_vae.csv')
sgRNAs = data['sgRNA'].values
nucleotides = ['A', 'C', 'G', 'T']
vocab_size = len(nucleotides)
max_length = max(len(sgRNA) for sgRNA in sgRNAs)
int_to_char = {i: nucleotide for i, nucleotide in enumerate(nucleotides)}

# Import and set up ONT and OFFT prediction models
ont_model = crispr_ont()
ont_model.load_weights("crispr_ont.h5")

def freeze(model):
    for layer in model.layers:
        layer.trainable = False

# Freeze the prediction models
freeze(ont_model)

@tf.keras.utils.register_keras_serializable()
class VAEWithLoss(Model):
    def __init__(self, latent_dim, vocab_size, max_length, embedding_dim, lstm_units, dropout_rate, ont_loss_epoch=30, ont_loss_weight=1.0, num_samples=5, **kwargs): # ont_model
        super(VAEWithLoss, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.ont_loss_epoch = ont_loss_epoch
        self.ont_loss_weight = ont_loss_weight
        self.num_samples = num_samples
        self.current_epoch = 0
        # self.is_epoch_end = False
        # self.ont_losses_in_epoch =[]
        # self.avg_ont_losses = []
        
        # Encoder definition
        self.encoder = self.build_encoder(vocab_size, embedding_dim, lstm_units, dropout_rate, latent_dim)
        
        # Decoder definition
        self.decoder = self.build_decoder(latent_dim, max_length, vocab_size, dropout_rate)
    
    @staticmethod
    def build_encoder(vocab_size, embedding_dim, lstm_units, dropout_rate, latent_dim):
        inputs = layers.Input(shape=(None,))
        x = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)
        x = layers.LSTM(lstm_units[0], return_sequences=True)(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.LSTM(lstm_units[1])(x)
        x = layers.Dropout(dropout_rate)(x)
        z_mean = layers.Dense(latent_dim)(x)
        z_log_var = layers.Dense(latent_dim)(x)

        # reparameterization trick: z = mean + std_dev * epsilon
        z = layers.Lambda(lambda args: args[0] + tf.exp(0.5 * args[1]) * tf.keras.backend.random_normal(tf.shape(args[0])))([z_mean, z_log_var])
        return Model(inputs, [z_mean, z_log_var, z], name='encoder')
    
    @staticmethod
    def build_decoder(latent_dim, max_length, vocab_size, dropout_rate):
        decoder_inputs = layers.Input(shape=(latent_dim,))
        x = layers.Dense(64, activation='relu')(decoder_inputs)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.RepeatVector(max_length)(x)
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.LSTM(128, return_sequences=True)(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.TimeDistributed(layers.Dense(vocab_size, activation='softmax'))(x)
        return Model(decoder_inputs, outputs, name='decoder')
    
    def get_config(self):
        config = super(VAEWithLoss, self).get_config()
        config.update({
            "latent_dim": self.latent_dim,
            "vocab_size": self.vocab_size,
            "max_length": self.max_length,
            "embedding_dim": self.encoder.layers[1].output_dim,
            "lstm_units": [self.encoder.layers[2].units, self.encoder.layers[4].units],
            "dropout_rate": self.encoder.layers[3].rate,
            "ont_loss_epoch": self.ont_loss_epoch,
            "ont_loss_weight": self.ont_loss_weight,
            "num_samples": self.num_samples
        })
        return config    
    
    def set_current_epoch(self, epoch):
        self.current_epoch = epoch
    
    def ont_eval(self, sgRNAs_tensor):
        # Decode each sgRNA sequence and predict efficacy using tf.map_fn
        efficacies = tf.map_fn(
            lambda sgRNA: tf.py_function(func=self.predict_efficacy, inp=[sgRNA], Tout=tf.float32),
            sgRNAs_tensor,
            dtype=tf.float32
        )
        return efficacies

    def predict_efficacy(self, sgRNA_tensor):
        # Decode the sgRNA_tensor into a string
        sgRNA_string = sgRNA_tensor.numpy().decode('utf-8')

        # Make the data input for the ONT model prediction
        x_test = make_data('N' + sgRNA_string)
        efficacy_pred = ont_model.predict([x_test])
        return tf.convert_to_tensor(efficacy_pred, dtype=tf.float32)
    
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)

        # Generate sgRNA samples then compute on-target loss
        def decode_sequence(sequence):
            return ''.join([int_to_char[int(i)] for i in sequence])

        # Sample from the actual latent space
        random_latent_vectors = tf.random.normal(shape=(self.num_samples, self.latent_dim))
        generated_sgRNAs = self.decoder(random_latent_vectors)
        generated_sequences = tf.argmax(generated_sgRNAs, axis=-1)
        
        decoded_generated_sgRNAs = tf.map_fn(
            lambda seq: tf.py_function(func=decode_sequence, inp=[seq], Tout=tf.string),
            generated_sequences, dtype=tf.string)

        ont_efficacies = self.ont_eval(decoded_generated_sgRNAs)
        min_efficacy = tf.reduce_min(ont_efficacies)
        min_target_efficacy = 1.0  # Ground truth on-target efficacy
        ont_efficacy_loss = tf.reduce_mean(tf.square(min_efficacy - min_target_efficacy))
        # self.ont_losses_in_epoch.append(ont_efficacy_loss)

        # KL divergence loss
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss) * -0.5

        # Reconstruction loss
        reconstruction_loss = tf.keras.losses.sparse_categorical_crossentropy(inputs, reconstructed)
        reconstruction_loss *= self.max_length

        # Compute total loss
        total_loss = tf.reduce_mean(reconstruction_loss + kl_loss + self.ont_loss_weight * ont_efficacy_loss)
        self.add_loss(total_loss)
        return reconstructed

# Load architecture
with open('vae5_architecture.json', 'r') as json_file:
    json_config = json_file.read()

loaded_vae = tf.keras.models.model_from_json(json_config, custom_objects={'VAEWithLoss': VAEWithLoss})

print("Loading vae...")
# Build the model to ensure it's ready to load weights
loaded_vae.build(input_shape=(None, max_length))
loaded_vae.load_weights('best_vae5.weights.h5')

print("Successfully loaded vae!")
loaded_decoder = loaded_vae.decoder

#print("Loading decoder weights...")
#loaded_decoder.load_weights('vae6/best_decoder6.weights.h5')
freeze(loaded_decoder)
#print("Successfully loaded decoder weights!")


def generate_sgRNA(decoder, latent_dim, num_samples=1):
    random_latent_vectors = np.random.normal(size=(num_samples, latent_dim))
    generated_sequences = decoder.predict(random_latent_vectors)
    generated_sgRNAs = np.argmax(generated_sequences, axis=-1)

    def decode_sequence(sequence):
        return ''.join(int_to_char[i] for i in sequence)

    return [decode_sequence(seq) for seq in generated_sgRNAs]

def ont_eval(new_sgrnas):
    sgrna_efficacy_pairs = []
    for sgrna in new_sgrnas:
        x_test = make_data('N' + sgrna)
        efficacy_pred = ont_model.predict([x_test])
        sgrna_efficacy_pairs.append({'sgRNA': sgrna, 'efficacy': efficacy_pred})
    return sgrna_efficacy_pairs

# Generate 10 new sgRNAs
new_sgRNAs = generate_sgRNA(loaded_decoder, latent_dim=32, num_samples=1000)
sgrna_efficacy_pairs = ont_eval(new_sgRNAs)

# Save the list to a file
with open('sgRNA_efficacies_pairs5.txt', 'w') as file:
    for pair in sgrna_efficacy_pairs:
        file.write(f"{pair['sgRNA']}:{pair['efficacy']}\n")