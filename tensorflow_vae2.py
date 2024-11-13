#!/usr/bin/env python3

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from crispr_offt_prediction import crispr_offt
from crispr_ont_prediction import crispr_ont, make_data

# Import and set up ONT and OFFT prediction models
ont_model = crispr_ont()
offt_model = crispr_offt()

print("Loading weights for the prediction models")
ont_model.load_weights("crispr_ont.h5")
offt_model.load_weights("crispr_offt.h5")

# freeze the ont_model
for layer in ont_model.layers:
    layer.trainable = False

# Load the data
data = pd.read_csv('sgrna_hela_vae.csv')
sgRNAs = data['sgRNA'].values

# Define the vocabulary
nucleotides = ['A', 'C', 'G', 'T']
vocab_size = len(nucleotides)
max_length = max(len(sgRNA) for sgRNA in sgRNAs)

# Create a mapping from nucleotides to integers
char_to_int = {nucleotide: i for i, nucleotide in enumerate(nucleotides)}
int_to_char = {i: nucleotide for i, nucleotide in enumerate(nucleotides)}

# Convert sgRNAs to sequences of integers
def encode_sgRNA(sgRNA):
    return [char_to_int[char] for char in sgRNA]

encoded_sgRNAs = np.array([encode_sgRNA(sgRNA) for sgRNA in sgRNAs])

# Pad the sequences
padded_sgRNAs = tf.keras.preprocessing.sequence.pad_sequences(encoded_sgRNAs, maxlen=max_length, padding='post')

# Define VAE components
latent_dim = 32  # Dimension of the latent space

# Encoder
inputs = layers.Input(shape=(max_length,))
x = layers.Embedding(input_dim=vocab_size, output_dim=32)(inputs)  # Increased embedding dimension
x = layers.LSTM(128, return_sequences=True)(x)  # Increased LSTM units
x = layers.Dropout(0.3)(x)  # Add dropout layer with 30% dropout rate
x = layers.LSTM(64)(x)  # Increased LSTM units
x = layers.Dropout(0.3)(x)  # Add dropout layer with 30% dropout rate
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)

# Sampling function
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Latent space
z = layers.Lambda(sampling)([z_mean, z_log_var])

# Decoder
decoder_inputs = layers.Input(shape=(latent_dim,))
x = layers.Dense(64, activation='relu')(decoder_inputs)  # Increased dense layer size
x = layers.Dropout(0.3)(x)  # Add dropout layer with 30% dropout rate
x = layers.RepeatVector(max_length)(x)
x = layers.LSTM(64, return_sequences=True)(x)  # Increased LSTM units
x = layers.Dropout(0.3)(x)  # Add dropout layer with 30% dropout rate
x = layers.LSTM(128, return_sequences=True)(x)  # Increased LSTM units
x = layers.Dropout(0.3)(x)  # Add dropout layer with 30% dropout rate

outputs = layers.TimeDistributed(layers.Dense(vocab_size, activation='softmax'))(x)

# Custom layer for KL divergence and loss calculation
class VAEWithLoss(Model):
    def __init__(self, encoder, decoder, ont_model, efficacy_loss_weight=1.0, **kwargs):
        super(VAEWithLoss, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.ont_model = ont_model
        self.efficacy_loss_weight=efficacy_loss_weight
        self.current_epoch = 0

    def set_current_epoch(self, epoch):
        self.current_epoch = epoch  # Method to update the current epoch
    
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
        efficacy_pred = self.ont_model.predict([x_test])
        return tf.convert_to_tensor(efficacy_pred, dtype=tf.float32)


    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)

        # Generate new sgRNAs by sampling from the latent space
        num_samples = 5
        random_latent_vectors = tf.random.normal(shape=(num_samples, latent_dim))
        generated_sgRNAs = self.decoder(random_latent_vectors)

        # Convert the generated sgRNAs to nucleotide sequences
        generated_sequences = tf.argmax(generated_sgRNAs, axis=-1)

        # Function to decode the index sequences back to nucleotide characters
        def decode_sequence(sequence):
            return ''.join([int_to_char[int(i)] for i in sequence])

        # Decode all generated sequences
        decoded_generated_sgRNAs = tf.map_fn(
            lambda seq: tf.py_function(func=decode_sequence, inp=[seq], Tout=tf.string),
            generated_sequences, dtype=tf.string)

        # Use tf.print to output the generated sgRNAs
        # print("size: ", tf.size(decoded_generated_sgRNAs))
        # tf.print("Generated sgRNAs:", decoded_generated_sgRNAs, summarize=-1)
        ont_efficacies = self.ont_eval(decoded_generated_sgRNAs)
        # tf.print("\n efficacies: ", ont_efficacies)
        min_efficacy = tf.reduce_min(ont_efficacies)

        # Compute minimal sgRNA ont efficacy loss
        min_target_efficacy = 0.7
        ont_efficacy_loss = tf.reduce_mean(tf.square(min_efficacy - min_target_efficacy))


        # Compute the KL divergence
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss) * -0.5

        # Compute reconstruction loss
        reconstruction_loss = tf.keras.losses.sparse_categorical_crossentropy(inputs, reconstructed)
        reconstruction_loss *= max_length


        # Total loss
        total_loss = tf.reduce_mean(reconstruction_loss + kl_loss) if self.current_epoch < 30 else tf.reduce_mean(reconstruction_loss + kl_loss + self.efficacy_loss_weight * ont_efficacy_loss)
        self.add_loss(total_loss)

        return reconstructed

class UpdateEpochCallback(tf.keras.callbacks.Callback):
    def __init__(self, vae_model):
        super(UpdateEpochCallback, self).__init__()
        self.vae_model = vae_model

    def on_epoch_begin(self, epoch, logs=None):
        # Update the current epoch in the VAEWithLoss model
        self.vae_model.set_current_epoch(epoch)



# Create the encoder and decoder models
encoder_model = Model(inputs, [z_mean, z_log_var, z], name='encoder')
decoder_model = Model(decoder_inputs, outputs, name='decoder')

# Create the VAE model
vae = VAEWithLoss(encoder_model, decoder_model, ont_model)
update_epoch_callback = UpdateEpochCallback(vae)


# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Try lower or higher learning rates
vae.compile(optimizer=optimizer)

# Define callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',  # You can also monitor 'val_loss' if you have a validation set
    patience=5,      # Number of epochs with no improvement to stop training
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    filepath='best_vae_model2.keras',  # Filepath to save the best model
    monitor='val_loss',                # Monitoring loss for saving the model
    save_best_only=True,           # Save only the model with the best performance
    verbose=1
)


# Train the model with the callbacks
vae.fit(
    padded_sgRNAs, padded_sgRNAs,
    epochs=100,
    batch_size=64,
    validation_split=0.2,  # 20% of the data used for validation
    callbacks=[early_stopping, model_checkpoint, update_epoch_callback]
)
decoder_model.save_weights('decoder2.weights.h5')


# Generate new sgRNAs
def generate_sgRNA(decoder_model, latent_dim, num_samples=1):
    random_latent_vectors = np.random.normal(size=(num_samples, latent_dim))
    generated_sequences = decoder_model.predict(random_latent_vectors)
    generated_sgRNAs = np.argmax(generated_sequences, axis=-1)

    # Convert back to nucleotide sequences
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
new_sgRNAs = generate_sgRNA(decoder_model, latent_dim, num_samples=100)
sgrna_efficacy_pairs = ont_eval(new_sgRNAs)

# Save the list to a file
with open('sgRNA_efficacies_pairs2.txt', 'w') as file:
    for pair in sgrna_efficacy_pairs:
        file.write(f"{pair['sgRNA']}:{pair['efficacy']}\n")

print("Generated sgRNAs:", new_sgRNAs)
