#!/usr/bin/env python3

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import wandb
import random
from wandb.integration.keras import WandbCallback

from crispr_offt_prediction import crispr_offt
from crispr_ont_prediction import crispr_ont, make_data
import os

os.makedirs("vae6", exist_ok=True)
wandb.login(key="628545f3ecebbb741774074b3331ffdb3e4ad1fd")

# Initialize Weights & Biases
wandb.init(
    project='sgRNA-VAE6',
    entity='nagydavid02-bme',
    config={
        "latent_dim": 32,
        "embedding_dim": 32,
        "lstm_units": [128, 64],
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 350,
        "ont_loss_epoch": 0,  # from which epoch should the ont efficacy be included in the total loss
        "ont_loss_weight": 5.0,
        "offt_loss_weight": 1.0,
        "num_samples_in_vae": 5,
        "patience": 20,
        "validation_split": 0.2,
        "shuffle": True
    }
)
config = wandb.config

# Import and set up ONT and OFFT prediction models
ont_model = crispr_ont()
offt_model = crispr_offt()
ont_model.load_weights("crispr_ont.h5")
offt_model.load_weights("crispr_offt.h5")

def freeze(model):
    for layer in model.layers:
        layer.trainable = False

# Freeze the prediction models
freeze(ont_model)
freeze(offt_model)

# Define the nucleotide pair encoding dictionary for off-target profile prediction
pair_encoding = {
    "AA": 0, "AC": 1, "AG": 2, "AT": 3,
    "CA": 4, "CC": 5, "CG": 6, "CT": 7,
    "GA": 8, "GC": 9, "GG": 10, "GT": 11,
    "TA": 12, "TC": 13, "TG": 14, "TT": 15
}

def generate_offt_sequence(sequence):
    """Generate a similar sequence with 1-2 character changes."""
    sequence = list(sequence)  # Convert to list for mutability
    mutation_count = random.choice([1, 2])  # Randomly choose 1 or 2 mutations
    for _ in range(mutation_count):
        idx = random.randint(0, len(sequence) - 1)  # Random position in the sequence
        original_base = sequence[idx]
        new_base = random.choice([base for base in 'ACTG' if base != original_base])
        sequence[idx] = new_base  # Apply mutation
    return ''.join(sequence)

# Function to encode an target-sgRNA - OT pair
def encode_pairwise(sgrna, OT):
    encoded_sequence = []
    for s, d in zip(sgrna, OT):
        pair = s + d
        encoded_sequence.append(pair_encoding[pair])  # Convert each integer to a string
    return np.array([encoded_sequence])

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
encoded_sgRNAs = np.array([[char_to_int[char] for char in sgRNA] for sgRNA in sgRNAs])
padded_sgRNAs = tf.keras.preprocessing.sequence.pad_sequences(encoded_sgRNAs, maxlen=max_length, padding='post')

# Decoding function for sgRNA
def decode_sgRNA_sequence(sequence):
    return ''.join([int_to_char[int(i)] for i in sequence])

def generate_and_evaluate_sgRNAs(decoder, latent_dim, num_samples):
    random_latent_vectors = np.random.normal(size=(num_samples, latent_dim))
    generated_sequences = decoder.predict(random_latent_vectors)
    generated_sgRNAs = np.argmax(generated_sequences, axis=-1)

    # Decode and evaluate efficacies
    decoded_sgRNAs = [decode_sgRNA_sequence(seq) for seq in generated_sgRNAs]
    sgrna_ont_offt_triples = []
    for sgrna in decoded_sgRNAs:
        # on-target prediction
        x_test = make_data('N' + sgrna)
        ont_efficacy_pred = ont_model.predict([x_test])

        # off-target generation and prediction
        offt_seq = generate_offt_sequence(sgrna)
        encoded_features = encode_pairwise(sgrna, offt_seq)
        offt_efficacy_pred = offt_model.predict(encoded_features)
        sgrna_ont_offt_triples.append((sgrna, ont_efficacy_pred[0][0], offt_efficacy_pred))
    return sgrna_ont_offt_triples

# Custom layer for KL divergence and loss calculation
@tf.keras.utils.register_keras_serializable()
class VAEWithLoss(Model):
    def __init__(self, latent_dim, vocab_size, max_length, embedding_dim, lstm_units, dropout_rate, ont_loss_epoch=30, ont_loss_weight=1.0, offt_loss_weight=1.0, num_samples=5, **kwargs): # ont_model
        super(VAEWithLoss, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.ont_loss_epoch = ont_loss_epoch
        self.ont_loss_weight = ont_loss_weight
        self.offt_loss_weight = offt_loss_weight
        self.num_samples = num_samples
        self.ont_losses_in_epoch = []
        self.offt_losses_in_epoch = []
        self.vae_losses_in_epoch = []
        
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
            "offt_loss_weight": self.offt_loss_weight,
            "num_samples": self.num_samples
        })
        return config

    def predict_efficacy(self, sgRNA_tensor):
        # Decode the sgRNA_tensor into a string
        sgrna = sgRNA_tensor.numpy().decode('utf-8')

        # Make the data input for the ONT model prediction
        x_test = make_data('N' + sgrna)
        ont_efficacy_pred = ont_model.predict([x_test])

        # Generate offt seq for OFFT model prediction
        offt_seq = generate_offt_sequence(sgrna)
        encoded_features = encode_pairwise(sgrna, offt_seq)
        offt_efficacy_pred = offt_model.predict(encoded_features)
    
        ont_tensor = tf.convert_to_tensor(ont_efficacy_pred, dtype=tf.float32)
        offt_tensor = tf.convert_to_tensor(offt_efficacy_pred, dtype=tf.float32)

        ont_flat = tf.reshape(ont_tensor, [-1])
        offt_flat = tf.reshape(offt_tensor, [-1])

        combined_tensor = tf.concat([ont_flat, offt_flat], axis=0)
        return combined_tensor
     
    def _append_losses_in_epoch(self, ont_loss, offt_loss, vae_loss):
        self.ont_losses_in_epoch.append(ont_loss.numpy())  # Executed outside the graph, so .numpy() works here
        self.offt_losses_in_epoch.append(offt_loss.numpy())
        self.vae_losses_in_epoch.append(vae_loss.numpy())

    
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)

        # Sample from the actual latent space
        random_latent_vectors = tf.random.normal(shape=(self.num_samples, self.latent_dim))
        generated_sgRNAs = self.decoder(random_latent_vectors)
        generated_sequences = tf.argmax(generated_sgRNAs, axis=-1)

        decoded_generated_sgRNAs = tf.map_fn(lambda seq: tf.py_function(func=decode_sgRNA_sequence, inp=[seq], Tout=tf.string),generated_sequences, fn_output_signature=tf.string)
        ont_offt_combined = tf.map_fn(lambda sgRNA: tf.py_function(func=self.predict_efficacy, inp=[sgRNA], Tout=tf.float32), decoded_generated_sgRNAs, fn_output_signature=tf.float32)
        ont_length = 1  # ONT model predicts one value
        offt_length = 2  # OFFT model predicts 2 values

        # Slice the combined tensor
        ont_efficacies = ont_offt_combined[:, :ont_length]  # First value(s) for ont
        offt_efficacies = ont_offt_combined[:, ont_length:ont_length + offt_length]  # Remaining values for offt

        # Ground truth for on-target efficacy
        min_ont_target_efficacy = 1.0  
        min_ont_efficacy = tf.reduce_min(ont_efficacies)
        ont_efficacy_loss = tf.reduce_mean(tf.square(min_ont_efficacy - min_ont_target_efficacy))
        
        # Ground truth for off-target efficacy (1 for on-target, 0 for off-target)
        offt_target = tf.constant([[1.0, 0.0]], dtype=tf.float32)
        offt_targets = tf.repeat(offt_target, tf.shape(offt_efficacies)[0], axis=0)  # Shape (batch_size, 2)

        # Ensure offt_efficacies has a known shape
        offt_efficacies = tf.identity(offt_efficacies)  # Convert to a concrete tensor
        offt_efficacies = tf.reshape(offt_efficacies, [-1, 2])  # Ensure it's of shape (batch_size, 2)

        # Calculate BCE loss between predicted offt_efficacies and the target
        offt_efficacy_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(offt_targets, offt_efficacies))

        # KL divergence loss
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss) * -0.5

        # Reconstruction loss
        reconstruction_loss = tf.keras.losses.sparse_categorical_crossentropy(inputs, reconstructed)
        reconstruction_loss *= self.max_length

        # Compute default vae loss
        vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss) * 1.0
        # Using tf.py_function to append ont and offt efficacy_loss to a list outside the symbolic graph
        tf.py_function(self._append_losses_in_epoch, [ont_efficacy_loss, offt_efficacy_loss, vae_loss], [])
        
        # Compute total loss
        total_loss = tf.reduce_mean(reconstruction_loss + kl_loss + self.ont_loss_weight * ont_efficacy_loss) # + self.offt_loss_weight * offt_efficacy_loss
        self.add_loss(total_loss)
        return reconstructed

# Create the VAE model
vae = VAEWithLoss(
    latent_dim=config.latent_dim,
    vocab_size=vocab_size,
    max_length=max_length,
    embedding_dim=config.embedding_dim,
    lstm_units=config.lstm_units,
    dropout_rate=config.dropout_rate,
    ont_loss_epoch=config.ont_loss_epoch,
    ont_loss_weight=config.ont_loss_weight,
    offt_loss_weight=config.offt_loss_weight,
    num_samples=config.num_samples_in_vae
)
print(vae.summary())
# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
vae.compile(optimizer=optimizer)


class LossCallback(tf.keras.callbacks.Callback):
    def __init__(self, vae_model):
        super(LossCallback, self).__init__()
        self.vae_model = vae_model

    def on_epoch_end(self, epoch, logs=None):
        # Update the model's current_epoch attribute
        avg_ont_loss = sum(self.vae_model.ont_losses_in_epoch) / len(self.vae_model.ont_losses_in_epoch)
        avg_offt_loss = sum(self.vae_model.offt_losses_in_epoch) / len(self.vae_model.offt_losses_in_epoch)
        avg_vae_loss = sum(self.vae_model.vae_losses_in_epoch) / len(self.vae_model.vae_losses_in_epoch)
  
        log_dict = {
            "avg_ont_loss": avg_ont_loss,
            "avg_offt_loss": avg_offt_loss,
            "avg_vae_loss": avg_vae_loss
            }
        wandb.log(log_dict, step=epoch)

        self.vae_model.ont_losses_in_epoch.clear()
        self.vae_model.offt_losses_in_epoch.clear()
        self.vae_model.vae_losses_in_epoch.clear()


lossCallbacks = LossCallback(vae)
model_checkpoint = ModelCheckpoint(filepath='vae6/best_vae6_ONT_WEIGHT.weights.h5', monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min', verbose=1)
wandb_callback = WandbCallback(monitor='val_loss', save_model=False)

# Train the model
vae.fit(
    padded_sgRNAs, padded_sgRNAs,
    epochs=config.epochs,
    batch_size=config.batch_size,
    validation_split=config.validation_split,
    shuffle=config.shuffle,
    callbacks=[model_checkpoint, lossCallbacks, wandb_callback]
)

wandb.finish()

# save vae in last epoch for test
vae.save_weights('vae6/vae6_ONT_WEIGHT_epoch.weights.h5')

# Save the architecture
with open('vae6/vae6_architecture.json', 'w') as json_file:
    json_file.write(vae.to_json())

# # Load architecture
with open('vae6/vae6_architecture.json', 'r') as json_file:
     json_config = json_file.read()

loaded_vae = tf.keras.models.model_from_json(json_config, custom_objects={'VAEWithLoss': VAEWithLoss})

print("Loading vae...")
# # Build the model to ensure it's ready to load weights
loaded_vae.build(input_shape=(None, max_length))
loaded_vae.load_weights('vae6/best_vae6_ONT_WEIGHT.weights.h5')

print("Successfully loaded vae! :o")

loaded_encoder = loaded_vae.encoder
loaded_decoder = loaded_vae.decoder

# print("Saving encoder and decoder weights...")
loaded_encoder.save_weights('vae6/best_encoder6.weights.h5')
loaded_decoder.save_weights('vae6/best_decoder6.weights.h5')

print("Loading decoder weights...")
loaded_decoder.load_weights('vae6/best_decoder6.weights.h5')
freeze(loaded_decoder)
print("Successfully loaded decoder weights! :o")

# Generate 100 new sgRNAs
sgrna_ont_efficacy_pairs = generate_and_evaluate_sgRNAs(loaded_decoder, config.latent_dim, num_samples=1000)

# Save the list to a file
with open('vae6/sgRNA_efficacies_ONT_WEIGHT_pairs6.txt', 'w') as file:
    for sgrna, ont_eff, offt_eff in sgrna_ont_efficacy_pairs:
        file.write(f"{sgrna},{ont_eff},{offt_eff}\n")